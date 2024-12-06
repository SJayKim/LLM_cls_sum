# llama_api.py
import os, sys
## 상위 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.prompt.generate_prompt import generate_classify_prompt, generate_summarize_prompt

from dotenv import load_dotenv
import torch
from transformers import pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables
load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_TOKEN")

app = FastAPI()

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

class SummarizeRequest(BaseModel):
    text: str

class ClassifyRequest(BaseModel):
    text: str

class Message(BaseModel):
    role: str
    content: str

class GenerateResponseRequest(BaseModel):
    messages: list[Message]



@app.post("/summarize_document/")
async def summarize_document(request_body: SummarizeRequest):
    generated_prompt = generate_summarize_prompt(request_body.text, "summarize")
    messages = [{"role": generated_prompt["role"], "content": generated_prompt["message"]}]
    try:
        outputs = pipe(
            messages,
            max_new_tokens=256,
        )
        response_message = outputs[0]["generated_text"]
        return {"response": response_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify_topic/")
async def classify_topic(request_body: ClassifyRequest):
    generated_prompt = generate_classify_prompt(request_body.text, "classify")
    messages = [{"role": generated_prompt["role"], "content": generated_prompt["message"]}]
    try:
        outputs = pipe(
            messages,
            max_new_tokens=256,
        )
        response_message = outputs[0]["generated_text"]
        return {"response": response_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_response/")
async def generate_response(request_body: GenerateResponseRequest):
    messages = [{"role": msg.role, "content": msg.content} for msg in request_body.messages]
    try:
        outputs = pipe(
            messages,
            max_new_tokens=256,
        )
        response_message = outputs[0]["generated_text"]
        return {"response": response_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)