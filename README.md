# LLM_cls_sum
Classifier and summarization using LLM

# LLM_cls_sum

Classifier and summarization using LLM (LLAMA 3.2)

## 개요

이 프로젝트는 LLAMA 3.2 모델을 사용하여 텍스트 데이터를 주제별로 분류하고 요약하는 기능을 제공합니다. FastAPI를 사용하여 웹 API로 제공되며, Hugging Face API를 통해 모델을 호출합니다.

## 기능

- 텍스트 데이터의 주제 분류
- 텍스트 데이터의 요약
- FastAPI를 사용한 웹 API 제공

## 설치

### 필수 조건

- Python 3.8 이상
- 가상 환경 (권장)

### 설치 방법

1. 저장소를 클론합니다:

    ```bash
    git clone https://github.com/yourusername/LLM_cls_sum.git
    cd LLM_cls_sum
    ```

2. 가상 환경을 생성하고 활성화합니다:

    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python -m venv venv
    source venv/bin/activate
    ```

3. 필요한 패키지를 설치합니다:

    ```bash
    pip install -r requirements.txt
    ```

4. `.env` 파일을 생성하고 Hugging Face API 토큰을 추가합니다:
    - 추가적으로 HuggingFace 에서 LLAMA 3.2 모델 사용 허가를 받아야 합니다.
    ```plaintext
    HUGGINGFACE_API_TOKEN=your_huggingface_api_token
    ```


## 사용 방법

### FastAPI 서버 실행

FastAPI 서버를 실행하여 웹 API를 제공합니다:

```bash
uvicorn llama_api:app --host 0.0.0.0 --port 8001 --reload