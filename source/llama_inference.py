import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a keyword generating machine that generates keyword related to the movie"},
    {"role": "user", "content": "What keywords can you provide me for the movie Harry Potter"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
response_message = outputs[0]["generated_text"][-1]["content"]

print("=====Response=====")
print(response_message)
