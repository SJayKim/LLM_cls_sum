
import json
import os
from pprint import pprint

with open ("./prompt_configure.json", "r") as f:
    prompt_configure = json.load(f)

# pprint(prompt_configure)
def generate_prompt(input_text: str, prompt_type :str = None) -> str:
    if prompt_type not in ["classify", "summarize"]:
        raise ValueError("Invalid prompt type. Choose between 'classify' and 'summarize'.")
    current_prompt = prompt_configure[prompt_type]

    role = current_prompt["role"]

    main_prompt = f'''
    {current_prompt["header"]}
    '''

    if current_prompt["examples"]:
        example_prompt = f'''
        '''
        for idx, example in enumerate(current_prompt["examples"], start=1):
            example_prompt += f'''
    Example input{idx}: {example[0]}

    Example output{idx}: {example[1]}
    '''
        main_prompt += example_prompt

    main_prompt += f'''
    Main input text: {input_text}
    {current_prompt["footer"]}
        '''    

    output_message = {
        'role': role,
        'message': main_prompt
    }

    return output_message

if __name__ == "__main__":

    ## Test the function
    input_text = '''배관에서 영감을 받은 인공 심장
    '''
    prompt_type = "summarize"

    result = generate_prompt(input_text, prompt_type)
    print(result["message"])