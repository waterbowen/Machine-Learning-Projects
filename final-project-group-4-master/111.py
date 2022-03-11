import torch

a = torch.tensor([3,3,3])
b = torch.tensor([3,3,2])
print((a ==b).sum() )
