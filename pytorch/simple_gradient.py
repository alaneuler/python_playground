import torch

x = torch.tensor(0.5, requires_grad=True)
y = torch.tensor(0.8, requires_grad=True)
v = x * y
v.retain_grad()
w = torch.log(v)
w.backward()
print(x.grad, y.grad, v.grad)
