from transformers import AutoTokenizer, AutoModelForCausalLM
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from source.utils import Logger
import argparse

logger = Logger(log_file_path = './logs/model_download.log').get_logger()

def main(args):
    # 모델 이름 및 저장 경로 설정
    model_name = args.model_name
    save_directory = f"./model/{model_name}"
    logger.info(f"current model name: {model_name}")
    logger.info(f"save directory: {save_directory}")

    # 모델과 토크나이저 다운로드 및 저장
    logger.info("Downloading model and tokenizer...")
    try: 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Error while downloading tokenizer: {e}")
        return
    
    try:
        tokenizer.save_pretrained(save_directory)
    except Exception as e:
        logger.error(f"Error while saving tokenizer: {e}")
        return
    
    logger.info(f"Tokenizer saved to {save_directory}")


    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")  # GPU 자동 매핑
    except Exception as e:
        logger.error(f"Error while downloading model: {e}")
        return
    
    try:
        model.save_pretrained(save_directory)
    except Exception as e:
        logger.error(f"Error while saving model: {e}")
        return

    logger.info(f"Done, Model saved to {save_directory}, Model name: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
        
    args = parser.parse_args()
    print(args)

    main(args)

