
def generate_classify_prompt(input_text: str) -> str:
    role = "You are an expert of topic classification. Classify the given text into one of the following categories: Politics, Sports, Technology, Entertainment, Business, Health, Science, Education, or Other." 
    

    return f"Classify the topic of the following text: {input_text}"

def generate_summarize_prompt(input_text: str) -> str:
    role = "You are a summarizer. Summarize the given text in 1-2 sentences."
    

    return f"Summarize the following text: {input_text}"
