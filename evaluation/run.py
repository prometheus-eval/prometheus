import torch
from transformers import AutoTokenizer, LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("kaist-ai/Prometheus-13b-v1.0", device_map="auto")

input_text = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
[orig_instruction]

###Response to evaluate:
[orig_response]

###Reference Answer (Score 5):
[orig_reference_answer]

###Score Rubrics:
[[orig_criteria]]
Score 1: [orig_score1_description]
Score 2: [orig_score2_description]
Score 3: [orig_score3_description]
Score 4: [orig_score4_description]
Score 5: [orig_score5_description]

###Feedback: 
"""
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids, sample=True, temperature=1.0, top_p=0.9, max_new_tokens=256, repetition_penalty=1.03)
print(tokenizer.decode(outputs[0]))