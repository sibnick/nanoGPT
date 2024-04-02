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
        # if use_ordering:
        #     self.dropout2_0 = nn.Dropout(0)
        #     self.dropout2 = nn.Dropout(0.5)
        # else:
        #     self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(9216, num_features)
        self.fc2 = nn.Linear(num_features, 10)
        assert self.num_features % chunk_num == 0
        chunk = torch.ones((1, int(num_features/chunk_num)))
        chunks = []
        decay = initial_decay
        for n in range(self.chunk_num):
            # if n != 0:
            #     chunks.append(chunk * 1e-5)
            # else:
            #     chunks.append(chunk * 1e-6)
            chunks.append(chunk * decay)
            decay *= decay_factor
        self.decay = nn.Parameter(torch.concat(chunks, dim=1))
        self.decay.requires_grad = False
        chunks.reverse()
        self.decay_reverse = nn.Parameter(torch.concat(chunks, dim=1))
        self.decay_reverse.requires_grad = False
        print(self.decay.shape, self.calc_decay())


    def calc_decay(self):
        #import math
        return (self.fc2.weight * self.decay).abs_().sum() #+ (self.fc1.bias * self.decay_reverse).abs_().sum()

    def apply_features(self, features: int):
        self.trimed = True
        self.features = features
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
            x = self.trimed_fc1(x)
            x = F.relu(x)
            x = torch.concat([x, self.fc1.bias.data[self.features:].view(1, -1).repeat(x.shape[0], 1)], dim=1)
            #x = torch.concat([x, torch.zeros((x.shape[0], 128 - self.features), device=x.device)], dim=1)
            x = self.dropout2(x)
            x = self.fc2(x)
        else:
            x = self.fc1(x)
            x = F.relu(x)
            # if self.use_ordering:
            #     # chunks = [self.dropout2_0(x[:, 0:self.chunk_step]), self.dropout2(x[:, self.chunk_step:])]
            #     chunks = []
            #     start = 0
            #     for n in range(self.chunk_num):
            #         chunks.append(self.dropout2(x[:, start:(start + self.chunk_step)]))
            #         start += self.chunk_step
            #     x = torch.concat(chunks, dim=1)
            # else:
            #     x = self.dropout2(x)
            x = self.dropout2(x)
            x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output