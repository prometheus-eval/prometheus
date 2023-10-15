# Prometheus
An Evaluator LM that is open-source, offers reproducible evaluation, and inexpensive to use. Specifically designed for fine-grained evaluation on a customized score rubric, Prometheus is a good alternative for human evaluation and GPT-4 evaluation.

We'll update the code soon, stay-tuned!

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
