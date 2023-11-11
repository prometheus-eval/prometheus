import aiohttp
import requests
import json
from conversation import get_conv_template
from tqdm import tqdm
import argparse
import os
from transformers import AutoTokenizer
import copy
import re

def create(url: str, payload: dict):
    header = {"Content-Type": "application/json"}
    response = requests.post(url, headers=header, data=json.dumps(payload))
    if response.ok:
        response = json.loads(response.text)
        print(response)
        x = response['generated_text'].split('[RESULT]')
        if len(x)>=2:
            score = x[1].strip()
            if score not in ["1","2","3","4","5"]:
                print(x)
                return "Something went wrong.", "Something went wrong."
            feedback = x[0].strip()
        else:
            return "Something went wrong.", "Something went wrong."
    else:
        # print(url)
        # print(payload)
        print(f"{response.status_code}: {response.reason}")
        return "Something went wrong.","Something went wrong."
    return feedback, int(score)


def main(args):
    #url = "http://137.68.191.121:3000/generate"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",cache_dir="../../../hf_cache")
    url = args.server
    params = {
        "best_of": 1,
        "decoder_input_details": False,
        "details": False,
        "do_sample": True,
        "max_new_tokens": 256,
        "repetition_penalty": 1.03,
        "return_full_text": False,
        # "seed": 42,
        # "stop": ["[END]"],
        "temperature": 1.0,
        # "top_k": 10,
        "top_p": 0.9,
    }

    data = []
    with open(args.input_file_name, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    
    with open(args.output_file_name,'w') as file:
        for dialogs in tqdm(data):
            conv = get_conv_template("llama-2")
            conv.set_system_message("You are a fair evaluator language model.")
            conv.append_message(conv.roles[0], dialogs['instruction'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            # print(prompt)
            x = tokenizer(prompt,truncation=False)
            if(len(x['input_ids'])>3072):
                continue
            
            payload = {
                "inputs": prompt,
                "parameters": params,
            }
            
            result = copy.deepcopy(dialogs)
            result['prometheus_output'] = []
            result['prometheus_score'] = []

            for idx in range(3):         
                while True:
                    try:
                        response,score = create(url, payload)
                        if response != "Something went wrong.":
                            result['prometheus_output'].append(response)
                            result['prometheus_score'].append(score)
                            break
                    except:
                        continue

            file.write(json.dumps(result)+"\n")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_name",
        type=str,
        help="directory with json files to inference.")
    parser.add_argument(
        "--output_file_name",
        type=str,
        help="directory to save the inferenced json files.")
    parser.add_argument(
        "--server",
        type=str,
        help="inference server.")
    args = parser.parse_args()
    main(args)