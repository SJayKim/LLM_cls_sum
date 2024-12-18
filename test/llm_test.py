import os, sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(parent_dir)
sys.path.append(parent_dir)

from pprint import pprint

# llama_api.py
import os, sys
## 상위 경로 추가
from config.prompt.generate_prompt import generate_prompt
from source.utils import check_gpu_allocation

from dotenv import load_dotenv
import torch
from transformers import pipeline

import pandas as pd
from collections import defaultdict


def summarize_document(document: list):

    # print(f'for debug: {document}')
    messages = []
    for text in document:
        generated_prompt = generate_prompt(text, "summarize")
        message = [{"role": "system", "content": generated_prompt["role"]}, 
                    {"role": "user", "content": generated_prompt["message"]}]
        # print(f"debug: {message}")
        messages.append(message)
    # print(f"debug: {len(messages)}")
    try:
        outputs = pipe(
            messages,
            max_new_tokens=12800,
        )
        response_message = [output[0]["generated_text"][-1]["content"] for output in outputs] 
        return {"response": response_message}
    except Exception as e:
        raise Exception(str(e))
    


def classify_topic(document: list):
    messages = []
    for text in document:
        generated_prompt = generate_prompt(text, "classify")
        # print(generated_prompt["message"])
        message = [{"role": "system", "content": generated_prompt["role"]}, 
                    {"role": "user", "content": generated_prompt["message"]}]
        messages.append(message)
    try:
        outputs = pipe(
            messages,
            max_new_tokens=12800,
        )
        response_message = [output[0]["generated_text"][-1]["content"] for output in outputs]
        return {"response": response_message}

    except Exception as e:
        raise Exception(str(e))

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")

    ## 모델 parameter setting 및 모델 로드
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        do_sample=False,
        top_p = None,
        temperature = None
        # trust_remote_code = True
    )

    ## 현재 GPU 스펙 및 allocation 현황 조회
    check_gpu_allocation()

    ## 데이터 로드
    data_path = "C:/Users/cyon1/Desktop/LLM_cls_sum/data/article_samples.xlsx"
    data = pd.read_excel(data_path)

    ## 데이터 전처리
    original_documents = data["Articles"].tolist()

    ## 요약 및 분류
    summarized_documents = summarize_document(original_documents)
    classified_documents = classify_topic(original_documents)

    ## 결과 저장
    save_file = {
        "original_documents": original_documents,
        "summarized_documents": summarized_documents["response"],
        "classified_documents": classified_documents["response"]
    }

    df = pd.DataFrame(save_file)
    df.to_excel("C:/Users/cyon1/Desktop/LLM_cls_sum/data/LLAMA3B_result.xlsx", index=False)
