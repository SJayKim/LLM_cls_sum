


def generate_classify_prompt(input_text: str) -> str:
    
    role = "You are an expert of topic classification. Classify the given text into one of the following categories: Politics, Sports, Technology, Entertainment, Business, Health, Science, Education, or Other. Do not return any other description except for the categories and corresponding weights." 
    
    input_example1 = '''
    '''
    output_example1 = '''
    '''
    input_example2 = '''
    '''
    output_example2 = '''
    '''
    
    main_prompt = '''
    '''

    output_message = {
        'role': role,
        'message': main_prompt
    }

    return output_message

def generate_summarize_prompt(input_text: str) -> str:
    role = "You are a summarizer. Summarize the given text. Do not"
    
    input_example1 = '''
    The government unveiled policies to combat climate change, focusing on renewable energy, emission regulations, and electric vehicle incentives, aiming for carbon neutrality by 2050.
    '''
    output_example1 = '''
    '''
    input_example2 = '''
    '''
    output_example2 = '''
    '''

    

    return f"Summarize the following text: {input_text}"
