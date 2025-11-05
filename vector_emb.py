import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity


class Emb2VectMLP(nn.Module):

    def __init__(self, vocab_size=50257, n=768):
        super().__init__()
        self.v_emb = nn.Parameter(torch.rand((vocab_size, n))/100)
        self.register_buffer("w", self.v_emb)

    def forward(self):
        x = torch.pow(self.v_emb @ self.v_emb.T, 2)
        x = x.abs().mean()
        x2 = torch.pow(self.v_emb, 2).sum(dim=1)
        return x, (1 - x2.mean()).abs()

model = Emb2VectMLP()
model = model.to("cuda").train()
model = torch.compile(model)
device_type = 'cuda'
# note: float16 data type will automatically use a GradScaler
ptdtype = torch.bfloat16
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

s1 = (model.v_emb[2]*model.v_emb[2]).sum()
print("s1: ", s1)
print("cossim: ", cosine_similarity(model.v_emb[None, 2], model.v_emb[None, 20]))

for iter in range(1000):
    # with ctx:
    iter += 1
    loss1, loss2 = model()
    if iter % 10 == 0:
        print("Loss: ", loss1.item(), " ", loss2.item())
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.save(model.state_dict(), "out/ckpt-emb.pt")
print("-------------------")
s1 = (model.v_emb[2]*model.v_emb[2]).sum()
print("s1: ", s1)
print("cossim: ", cosine_similarity(model.v_emb[None, 2], model.v_emb[None, 20]))

