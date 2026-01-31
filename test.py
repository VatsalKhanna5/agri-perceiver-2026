import torch
from accelerate import Accelerator

acc = Accelerator()
print("Local rank:", acc.process_index, "Device:", torch.cuda.current_device())

x = torch.randn(2048,2048).to(acc.device)
y = x @ x
print("Done on rank", acc.process_index)
