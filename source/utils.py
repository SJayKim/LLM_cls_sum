import torch
import logging
import os


# GPU 포팅 확인 코드
def check_gpu_allocation(logger = None):
    

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if logger:
                logger.info(f"Device {i}: {torch.cuda.get_device_name(i)} - Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            else:
                print(f"Device {i}: {torch.cuda.get_device_name(i)} - Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
    else:
        if logger:
            logger.info(f"CUDA is not available. The model is running on CPU.")
        else:
            print("CUDA is not available. The model is running on CPU.")


class Logger:
    def __init__(self, log_file_path="logs/llm.log"):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 파일 핸들러 설정
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        # 스트림 핸들러 설정
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # 포맷터 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # 핸들러 추가
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def get_logger(self):
        return self.logger
