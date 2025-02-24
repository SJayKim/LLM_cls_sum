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
    model_id = "microsoft/Phi-3.5-MoE-instruct"
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
    # data_path = "/workspace/LLM_cls_sum/data/article_samples.xlsx"
    # data = pd.read_excel(data_path)
    
    
    
    # logger.info("Data loaded==================")

    # ## 데이터 전처리
    # original_documents = data["Articles"].tolist()

    original_documents = '''
    
    1996년 월간 난세계 신년호에는 한국란 공인 등록사업 추진 과정과 제2회 한
국춘란 엽예품 전국대회, 제14회 일본춘란추계전시대회의 이모저모가 수록되
는 등 다양한 볼거리로 가득했다. 그 가운데에서도 ‘1996년 한국난계를 전망
한다.’ 라는 코너는 난계 주요 인사들의 1996년 한국 난계에 대한 그들의 전
망과 견해를 담고 있어 매우 흥미로운 부분이었다. 이들의 당시 난계에 대
한 전망과 오늘날 우리 난계가 어떤 모습으로 진행되어왔는지 비교해 보
는것도의미가 있을것같아 ‘10년 전 난세계’ 그 첫 번째 주제로 다루어
보았다.
‘1996년 한국난계를 말한다’ 코너에는 학계, 취미계, 업계, 해외 등
다양한 분야의 애란인들 의견을 싣고 있다.
먼저 학계의 이종석 교수(서울여대 원예학과)는 앞으로는 우리나라
에 자생하는 춘란과 한란의 종간 교접종 육종사업이 활성화되고, 무
늬종 춘란의 배양이 애란인들의 관심을 끌 것이라고 예견했다. 당시
10여 년에 걸친 연구계획이 있다고 밝혔던 이종석 교수의 소식을 반
갑게도 2005년 7월 7일자 조선일보 기사에서 만날 수 있었다. 바로
그가 이끄는 연구팀이 우리나라 야생화를 개량해 신품종으로 육성하
는 것에 성공했다는 내용과 특허출원한 교접종, 일명 ‘향기나는 춘한
2006년 새해를 맞아 월간 난세계에서는 국내 난계의 과거를 돌아보고,
또 현재까지 어떤 과정을 거쳐 발전해 왔는지 더듬어 보자는 의미에서
10년 전 본지에 수록된 중요한 이슈를 중심으로 재해석하는 ‘10년 전
난세계’ 라는 연재 코너를 마련했다. 편안하게 과거와 현재를 둘러보길
바란다.
“1996년
한국난계를전망한다.”
【10년前난세계 】
1996년 1월호
DLLI
란『미악(米岳)』 을 개발했다는 소식이었다. 물론 국
내 난계에서 교접종이나 배양종을 경시하는 경향이
있지만 새로운 품종 개발을 위해 10여년의 시간동안
연구에 매진했을 학계의 노력에 감탄하지 않을 수
없었다.
취미계의 조관식(한국춘란회 부회장)씨 나날이 늘
어만 가는 새로운 용어의 신품종들은 우리 난계를
활발히 발전시키기 위한 애란인들의 욕구에서 비롯
되었다면서 산채나 유통에 종사하는 애란인들과 잡
지사의 노력과 공로를 인정하고 격려하는 자세가 필
요하다고 말했다. 그로부터 10년, 실로 우리 난계에
는 국내외적으로 다양한 품종의 난이 출현했고 또
우리 난의 국제적인 입지도 높아졌다. 마찬가지로
난 관련 용어도 세분화되어 난을 제대로 안다는 자
체가 어려울 정도로 난에 관한 학문은 방대한 지식
의 보고가 되었다. 또한 애란인, 상인, 잡지사라는
세 가지 중요한 요소가 우리 난계를 떠받들고, 어느
하나에도 치우침 없이 사이좋게 공생하고 있다. 10
년 전이나 지금이나 세 가지 단체의 어우러짐은 아
무리 강조해도 지나칠 것이 없는 듯하다.
업계의 조휴진(한밭자생난원 대표)씨는 점점 늘어
나고 있는 애란인의 숫자를 보며 머지않아 일본난
계를 능가할 국내 난계의 모습에 대해 기대를 표했
다. 또한 업계의 한 사람으로서 난의 지나친 경제적
가치 중시 현상을 우려하며 난상인들은 진실로 난
을 중히 여기는 마음으로 거래하여야 하며, 애란인
역시 난의 환금성을 벗어난 애란심을 키워야 한다
고 역설했다. 그 후로 10년, 그의 전망대로 우리 난
이 일본 전국대회에서 대상을 차지하는 등 여러 면
에서 일본을 능가하며 발전해 왔다. 하지만 그의 우
려대로 난의 높은 경제적 가치 때문에 불미스런 일
도 적지 않게 발생했다. 도난, 사기 행위 등 난계의
어두운 면이 종종 드러나곤 했던 것이다. 지금도 그
러한 악재를 끊을 수 있는 해결책은 양심, 신용의 회
복밖에 없음을
애란인과 상인
모두 인식하고
있다. 난계의 양
심적인 상거래
질서가 하루빨
리국내난계에
정착되어 접착
제처럼 찰싹 달
라붙어 있기를
바란다.
일본 난계의
히라노(平野綬) 교수는 1996년 비로소 국내 난계에
등록제도가 발족된 것을 축하한다고 전하며 과거 일
본 한란계의 등록제도 실패 사례를 빗대어 관상가치
가 우수한 품종을 선별하여 등록 허가하는 것이 오
랫동안 감상할 수 있음과 동시에 그 가치를 보존할
수 있다고 시사했다. 우리 난계의 숙원이었던 난 등
록제도는 1996년 1월을 기점으로 10년이 지난 지금
까지 그 맥락을 잘 이어오고 있다. 등록사업이 본격
화되면서 이름만 들어도알수있는명품의 우수한
품종들이 국내뿐만 아니라 해외에서도 우수한 성적
을 거두는 등 한국난계의 위상을 여러모로 떨치고
있다. 과연 외국인들도 우리 난계의 발전가능성을
오래 전부터 점치고 있었던 듯하다.
1996년 우리 난계를 전망했던 각계 인사들의 글은
10년이 지난 지금에 와서 읽어보아도 전혀 손색이
없는 주제들이었다. 돌아보면 우리 난계가 기특하게
잘해온 것도 있고, 반면에 당시의 이상향에는 아직
못 미치는 문제도 있다. 그럼에도 여전히 희망적인
것은 우리 난계가 조금씩 그 이상향과 거리를 좁혀
가고 있다는 사실이다. 우리 애란인들이 정말 잘해
왔고 또 잘하고 있다는 의미가 아닐까.
글｜박선미
    '''
    
    
    
    # ## 요약 및 분류
    logger.info("Article summarizing==================")
    summarized_documents = summarize_document(original_documents)
    print(summarized_documents)
   
   
    # logger.info("Article classifying")
    # classified_documents = classify_topic(original_documents)

    # ## 결과 저장
    # save_file = {
    #     "original_documents": original_documents,
    #     "summarized_documents": summarized_documents["response"],
    #     "classified_documents": classified_documents["response"]
    # }

    # df = pd.DataFrame(save_file)
    # df.to_excel("/workspace/LLM_cls_sum/output/LLAMATULU70B_result.xlsx", index=False)

    print("Done!")