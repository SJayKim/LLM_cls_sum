import torch

class ModelConfig:
    def __init__(
        self,
        name="meta-llama/Llama-3.2-3B-Instruct",
        task="text-generation",
        size="3B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ):
        self.name = name
        self.task = task
        self.size = size
        self.torch_dtype = torch_dtype
        self.device_map = device_map


