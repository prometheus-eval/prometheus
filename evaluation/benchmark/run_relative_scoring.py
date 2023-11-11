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
        response = json.loads(response.text)[0]
        print(response)
        x = response['generated_text'].split('[RESULT]')
        
        if len(x)>=2:
            score = x[1].strip()
            print(score)
            if score not in ["1","2","3","4","5"]:
                return "Something went wrong.", "Something went wrong."
            feedback = x[0].strip()
        else:
            # print(response['generated_text'].split('[END]',1)[0])
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
        "top_p": 0.95,
        # "truncate": "null",
        # "typical_p": 0.99,
        # "watermark": False,
    }

    
    with open(os.path.join(args.input_file_name), 'r') as file:
        data = json.load(file)
    
    with open(args.output_file_name,'w') as file:
        for dialogs in tqdm(data):
            conv1 = get_conv_template("llama-2")
            # conv1.set_system_message(dialogs['sys_prompt'])
            conv1.append_message(conv1.roles[0], dialogs['chosen_instruction'])
            conv1.append_message(conv1.roles[1], None)
            prompt1 = conv1.get_prompt()

            conv2 = get_conv_template("llama-2")
            # conv2.set_system_message(dialogs['sys_prompt'])
            conv2.append_message(conv2.roles[0], dialogs['rejected_instruction'])
            conv2.append_message(conv2.roles[1], None)
            prompt2 = conv2.get_prompt()
            # print(prompt)
            x1 = tokenizer(prompt1,truncation=False)
            x2 = tokenizer(prompt2,truncation=False)
            if(len(x1['input_ids'])>3072) or (len(x2['input_ids'])>3072):
                continue
            
            payload1 = {
                "inputs": prompt1,
                "parameters": params,
            }
            payload2 = {
                "inputs": prompt2,
                "parameters": params,
            }
            # print(create(url, payload)["generated_text"])
            result = copy.deepcopy(dialogs)

            
            while True:
                while True:
                    try:
                        response1,score1 = create(url, payload1)
                        if response1 != "Something went wrong.":
                            result['chosen_output']= response1
                            result['chosen_score']=score1
                            break
                    except:
                        continue
            
                while True:
                    try:
                        response2,score2 = create(url, payload2)
                        if response2 != "Something went wrong.":
                            result['rejected_output']=response2
                            result['rejected_score']=score2
                            break
                    except:
                        continue
                if result['tie']==0:
                    if result['chosen_score'] > result['rejected_score']:
                        result['accuracy']=1
                    elif result['chosen_score']< result['rejected_score']:
                        result['accuracy']=0
                    else:
                        continue
                else:
                    if result['chosen_score'] == result['rejected_score']:
                        result['accuracy']=1
                    else:
                        result['accuracy']=0
                file.write(json.dumps(result)+"\n")
                break
                
    

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