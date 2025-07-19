source .venv/bin/activate
source .env.pbs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

# Reference Running: qsub train/sft_qwen.pbs
uid="$(date +%Y%m%d_%H%M%S)"
# base_model="Qwen/Qwen2.5-VL-7B-Instruct"
base_model="Qwen/Qwen2.5-VL-32B-Instruct"
dataset_name="ob11/visual-prm-training-data-v1-mc0.0-qwen-format"
epochs=2
micro_batch_size=1 # -> batch_size will be 64 if 8 gpus, per device batch size in single node; max this without OOM
gradient_accumulation_steps=1 # gradually increase first, requires more GPU memory but less than increasing micro_batch_size
lr=5e-6
max_steps=-1 # -> not used now
min_lr=0 # -> not used now
weight_decay=1e-4 # -> not used now
gpu_count=$(nvidia-smi -L | wc -l) # -> not used now
push_to_hub=false # -> not used now

accelerate launch --config_file=train/deepspeed_zero3_grad_ckpt.yaml \
    train/sft_qwen.py \
    --dataset_name ${dataset_name} \
    --model_name_or_path ${base_model} \
    --per_device_train_batch_size ${micro_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --output_dir training_outputs/sft-$(echo ${dataset_name} | rev | cut -d'-' -f1-5 | rev)-${base_model}-${uid} \
    --bf16 True \
    --torch_dtype bfloat16 \
    --num_train_epochs ${epochs} \
    --learning_rate ${lr} \
    --gradient_checkpointing=True \
    --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}' \