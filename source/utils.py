import torch

# GPU 포팅 확인 코드
def check_gpu_allocation():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)} - Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
    else:
        print("CUDA is not available. The model is running on CPU.")
