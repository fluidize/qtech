import torch
from rich import print

q_values = torch.tensor([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

actions = torch.tensor([0, 1, 2])

print(q_values.gather(1, actions.unsqueeze(1)))
print(q_values.max(1))


