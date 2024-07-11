import torch

t = torch.rand(183, 3)
torch.save(t, "test_vec.pt")