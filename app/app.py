# llama_api.py
import os
from dotenv import load_dotenv
import torch
from transformers import pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables
load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_TOKEN")

# # Set the Hugging Face API token
# os.environ["HUGGINGFACE_API_TOKEN"] = api_token

app = FastAPI()

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
    )

class Message(BaseModel):
    role: str
    content: str

class RequestBody(BaseModel):
    messages: list[Message]

@app.post("/summarize_document/")
async def summarize_document(request_body: RequestBody):
    messages = [{"role": msg.role, "content": msg.content} for msg in request_body.messages]
    try:
        outputs = pipe(
            messages,
            max_new_tokens=256,
        )
        response_message = outputs[0]["generated_text"][-1]["content"]
        return {"response": response_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clssify_topic/")
async def classify_topic(request_body: RequestBody):
    messages = [{"role": msg.role, "content": msg.content} for msg in request_body.messages]
    try:
        outputs = pipe(
            messages,
            max_new_tokens=256,
        )
        response_message = outputs[0]["generated_text"][-1]["content"]
        return {"response": response_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_response/")
async def generate_response(request_body: RequestBody):
    messages = [{"role": msg.role, "content": msg.content} for msg in request_body.messages]
    try:
        outputs = pipe(
            messages,
            max_new_tokens=256,
        )
        response_message = outputs[0]["generated_text"][-1]["content"]
        return {"response": response_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)