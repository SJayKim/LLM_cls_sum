from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":

    # 저장된 모델 경로 설정
    load_directory = "./saved_llama_model"

    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(load_directory)
    model = AutoModelForCausalLM.from_pretrained(load_directory, device_map="auto")

    # 입력 문장 정의
    input_text = "Explain the benefits of AI in healthcare."

    # 입력을 토큰화
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # GPU에서 실행

    # 모델을 사용해 텍스트 생성
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=1000,  # 생성 문장 최대 길이
        num_return_sequences=3,  # 생성 결과 수
        temperature=0.7,  # 생성 다양성 조절
        top_k=50,  # 상위 K개 토큰에서 샘플링
        top_p=0.9  # 누적 확률 기반 샘플링
    )

    # 결과 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated Text:\n", generated_text)
