import os, sys


from pprint import pprint

# llama_api.py
import os, sys
## 상위 경로 추가
# 현재 파일의 경로를 기준으로 상위 폴더 경로 추가
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from config.prompt.generate_prompt import generate_prompt
from source.utils import check_gpu_allocation

from dotenv import load_dotenv
import torch
from transformers import pipeline

import pandas as pd
from collections import defaultdict
import logging

logger = logging.getLogger("LlmLogger")


# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG 레벨 이상의 모든 로그 출력
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 로그 포맷 설정
    handlers=[
        logging.FileHandler("/workspace/LLM_cls_sum/logs/llm_test.log"),  # 로그를 파일에 저장
        logging.StreamHandler()  # 로그를 콘솔에 출력
    ]
)


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
    model_id = "allenai/Llama-3.1-Tulu-3-70B"
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
    logger.info("Model loaded==================")
    ## 현재 GPU 스펙 및 allocation 현황 조회
    logger.info("GPU info ==================")
    check_gpu_allocation(logger = logger)

    ## 데이터 로드
    data_path = "/workspace/LLM_cls_sum/data/article_samples.xlsx"
    data = pd.read_excel(data_path)
    logger.info("Data loaded==================")

    ## 데이터 전처리
    original_documents = data["Articles"].tolist()

    ## 요약 및 분류
    logger.info("Article summarizing==================")
    summarized_documents = summarize_document(original_documents)
    logger.info("Article classifying")
    classified_documents = classify_topic(original_documents)

    ## 결과 저장
    save_file = {
        "original_documents": original_documents,
        "summarized_documents": summarized_documents["response"],
        "classified_documents": classified_documents["response"]
    }

    df = pd.DataFrame(save_file)
    df.to_excel("/workspace/LLM_cls_sum/output/LLAMATULU70B_result.xlsx", index=False)

    print("Done!")