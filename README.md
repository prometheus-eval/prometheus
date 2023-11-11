<h1 align="center">Prometheus</h1>

This is the official github repository for [Prometheus: Inducing Fine-grained Evaluation Capability in Language Models](https://arxiv.org/abs/2310.08491).


Prometheus is an evaluator LM that is open-source, offers reproducible evaluation, and inexpensive to use. Specifically designed for fine-grained evaluation on a customized score rubric, Prometheus is a good alternative for human evaluation and GPT-4 evaluation.


Citation:
```
@article{kim2023prometheus,
  title={Prometheus: Inducing Fine-grained Evaluation Capability in Language Models},
  author={Kim, Seungone and Shin, Jamin and Cho, Yejin and Jang, Joel and Longpre, Shayne and Lee, Hwaran and Yun, Sangdoo and Shin, Seongjin and Kim, Sungdong and Thorne, James and others},
  journal={arXiv preprint arXiv:2310.08491},
  year={2023}
}
```
### Setup

Install dependencies

```
pip install -r requirements.txt
```

### Input and Output Format of Prometheus

Prometheus is trained and inferenced using the following input prompt format. Note that you could fill in the instruction, response, reference answer, and score rubrics with your own data.

```
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Reference Answer (Score 5):
{orig_reference_answer}

###Score Rubrics:
[{orig_criteria}]
Score 1: {orig_score1_description}
Score 2: {orig_score2_description}
Score 3: {orig_score3_description}
Score 4: {orig_score4_description}
Score 5: {orig_score5_description}

###Feedback: 
```

Also, we use the following output format.

```
{orig_feedback}
[RESULT] {orig_score}
```

During inference, you could parse the score by splitting the number that is generated next to the \[RESULT\] phrase.

We also have the Feedback Collection dataset used to train Prometheus at this [link](https://huggingface.co/datasets/kaist-ai/Feedback-Collection).

### Training an Evaluator LM

To train a evaluator LM, you can use our code built upon the [llama-recipes](https://github.com/facebookresearch/llama-recipes) in the train directory.

```bash
torchrun 
    --nnodes NUM_SERVERS \
    --nproc_per_node NUM_GPUS \
    llama_finetuning.py \
    --model_name MODEL_NAME \
    --batch_size_training BATCH_SIZE
    --dist_checkpoint_root_folder ROOT_PATH_TO_SAVE_CKPTS \
    --dist_checkpoint_folder SUB_PATH_TO_SAVE_CKPTS \ 
    --dataset feedback_collection_freeform_dataset \
    --data_file DATA_FILE \
    --hf_cache_dir HF_CACHE_DIR \
    --num_epochs NUM_EPOCHS \
    --scheduler LR_SCHEDULER \
    --use_fast_kernel \
    --enable_fsdp \
    --pure_bf16 \
    --low_cpu_fsdp
```

As an alternative, you could consider implementing your custom code based on huggingface.


### Inference with an Evaluator LM

To quickly try Prometheus, use the basic template code in the inference directory.
Note that you should change the componenets (instruction, response, score rubric, reference answer) in the run.py file before running the code.

After filling in the input prompt template, you should apply the conversation template of Llama-2-Chat (not applying it might lead to unexpected behaviors).
You can find the conversation class at this [link](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py).
```
conv = get_conv_template("llama-2")
conv.set_system_message("You are a fair evaluator language model.")
conv.append_message(conv.roles[0], dialogs['instruction'])
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

x = tokenizer(prompt,truncation=False)
```

During our experiments, we used huggingface's [TGI](https://github.com/huggingface/text-generation-inference) for fast inference purposes. You need to acquire a server url and fill in the argument from the command below.

```
cd ./evaluation/benchmark
python3 run_absolute_scoring.py --input_file_name "./data/vicuna_eval.json" --output_file_name "./vicuna_eval_results.json" --server "http://1.1.1.1:3000/generate"
```

```
cd ./evaluation/benchmark
python3 run_relative_scoring.py --input_file_name "./data/mt_bench_human_judgement_eval.json" --output_file_name "./vicuna_eval_results.json" --server "http://1.1.1.1:3000/generate"
```