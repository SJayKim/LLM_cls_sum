import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) ## Add path to the parent directory

import torch
from transformers import pipeline

from config.model.model_config import ModelConfig


class Llm:
    def __init__(self, config: ModelConfig = None):
        '''
        Model config를 불러와 pipeline을 생성
        Config가 없을 경우 default config를 사용
        '''
        self.config = config if config else ModelConfig()
        self.pipe = pipeline(
            task=self.config.task,
            model=self.config.name,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device_map
        )

    def generate_response(self, role, prompt, max_tokens = 12800):
        messages = [
            {"role": role, "content": prompt},
        ]
        try:
            outputs = self.pipe(
                messages,
                max_new_tokens=max_tokens
            )
            response_message = outputs[0]["generated_text"][-1]["content"]
            return response_message
        
        except Exception as e:
            return str(e)
