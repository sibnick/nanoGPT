import torch
import torch.nn as nn
import torch.nn.functional as F


class MNSIT_OrderedNet(nn.Module):
    def __init__(self, use_ordering=True, initial_decay: float = 1e-6, decay_factor: float = 2, num_features: int = 128, chunk_num: int = 4) -> None:
        super(MNSIT_OrderedNet, self).__init__()
        self.trimed = False
        self.use_ordering = use_ordering
        self.num_features = num_features
        self.chunk_num = chunk_num
        self.chunk_step = int(self.num_features / self.chunk_num)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, num_features)
        self.fc2 = nn.Linear(num_features, 10)
        assert self.num_features % chunk_num == 0
        chunk = torch.ones((1, int(num_features/chunk_num)))
        chunks = []
        decay = initial_decay
        for n in range(self.chunk_num):
            chunks.append(chunk * decay)
            decay *= decay_factor
        self.decay = nn.Parameter(torch.concat(chunks, dim=1))
        self.decay.requires_grad = False
        chunks.reverse()
        self.decay_reverse = nn.Parameter(torch.concat(chunks, dim=1))
        self.decay_reverse.requires_grad = False

        #self.fc1.weight.data.mul_(initial_decay / self.decay.T)
        print(self.fc2.weight.shape)
        self.rework_count = 0


    def apply_decay(self):
        self.fc2.weight *= (1-self.decay)
    def calc_decay(self):
        return (self.fc2.weight * self.decay).abs_().sum()
        # tmp = (self.fc2.weight[:, 1:] / (1e-9 + self.fc2.weight[:, :-1]))
        # tmp = tmp * tmp
        # t = torch.arange(127, 0, step=-1, requires_grad=False).cuda()
        # return (tmp * t).abs().mean()

    def apply_features(self, features: int, fixed_threshold: float = 2.0, dynamic_threshold: float = 0.1):
        self.trimed = True
        self.features = features
        self.fixed_threshold = fixed_threshold
        self.dynamic_threshold = dynamic_threshold
        self.trimed_fc1 = self.trim_layer(self.fc1, features)

    def trim_layer(self, fc: nn.Linear, features: int) -> nn.Linear:
        weight = fc.weight
        bias = fc.bias
        fc = nn.Linear(weight.shape[1], features)
        fc.weight = nn.Parameter(weight[0:features, :])
        fc.bias = nn.Parameter(bias[0:features])
        fc.to(weight.device)
        return fc

    def weight_dist_by_chunk(self) -> list[float]:
        rs = []
        start = 0
        for n in range(self.chunk_num):
            rs.append(self.fc2.weight[:, start:(start + self.chunk_step)].abs().sum().detach().cpu().item())
            start += self.chunk_step
        return rs
    def bias_dist_by_chunk(self) -> list[float]:
        rs = []
        start = 0
        for n in range(self.chunk_num):
            rs.append(self.fc1.bias[start:(start + self.chunk_step)].abs().sum().detach().cpu().item())
            start += self.chunk_step
        return rs

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        if self.trimed:
            x1 = self.trimed_fc1(x)
            x1 = F.relu(x1)
            x1 = torch.concat([x1, self.fc1.bias.data[self.features:].view(1, -1).repeat(x1.shape[0], 1)], dim=1)
            #x = torch.concat([x, torch.zeros((x.shape[0], 128 - self.features), device=x.device)], dim=1)
            x1 = self.dropout2(x1)
            x1 = self.fc2(x1)
        else:
            x1 = self.fc1(x)
            x1 = F.relu(x1)
            x1 = self.dropout2(x1)
            x1 = self.fc2(x1)
        output = F.log_softmax(x1, dim=1)

        if self.trimed and self.fixed_threshold > 0:
            top2_output, _ = output.topk(2, dim=1)
            top2_output, _ = torch.sort(top2_output, dim=1)
            rate = top2_output[:, 0] / top2_output[:, 1]
            need_rework = rate < self.fixed_threshold
            rework_percent = 1.0 * need_rework.sum() / output.shape[0]
            if self.dynamic_threshold < rework_percent:
                _, need_rework = rate.topk(int(output.shape[0]*self.dynamic_threshold), largest=False)
            x2 = x[need_rework]
            x2 = self.fc1(x2)
            x2 = F.relu(x2)
            x2 = self.dropout2(x2)
            x2 = self.fc2(x2)
            output2 = F.log_softmax(x2, dim=1)
            output[need_rework] = output2
            self.rework_count += output2.shape[0]
        return output