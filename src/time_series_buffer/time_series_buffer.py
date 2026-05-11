import torch


class TimeSeriesBuffer:
    def __init__(self, num_envs, dim:int, max_size: int, stride: int, device='cpu'):
        self.buffer = torch.zeros((num_envs, max_size, dim), device=device)
        self.max_size = max_size
        self.stride = stride
        self.indices = torch.arange(self.max_size - 1, -1, -self.stride, device=device)
        self.batch_indices = torch.arange(num_envs, dtype=torch.long, device=device)
        self.write_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
    
    def add(self, value):
        value_squeezed = value.squeeze()
        self.buffer[self.batch_indices, self.write_indices, :] = value_squeezed
        self.write_indices = (self.write_indices + 1) % self.max_size

    def get(self):
        # Adjust indices to account for circular buffer position
        adjusted_indices = (self.indices.unsqueeze(0) + self.write_indices.unsqueeze(1)) % self.max_size
        return self.buffer[self.batch_indices.unsqueeze(1), adjusted_indices, :]

    def reset(self, reset_mask: torch.Tensor):
        self.buffer[reset_mask, :, :] = 0.0
        self.write_indices[reset_mask] = 0
