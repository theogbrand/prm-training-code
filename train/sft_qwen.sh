# Reference Running: bash train/sft.sh
# {'train_runtime': 5268.8407, 'train_samples_per_second': 0.949, 'train_steps_per_second': 0.119, 'train_loss': 0.1172730620391667, 'epoch': 5.0}
uid="$(date +%Y%m%d_%H%M%S)"
# base_model="Qwen/Qwen2.5-VL-3B-Instruct"
base_model="Qwen/Qwen2.5-VL-7B-Instruct"
# base_model="Qwen/Qwen2.5-VL-32B-Instruct"
# dataset_name="ob11/ai2d-prm-training-data-v0.4-pil"
dataset_name="ob11/visual-prm-training-data-v1-mc0.0-qwen-format"
lr=2e-5
epochs=2
micro_batch_size=1 # -> batch_size will be 64 if 8 gpus, per device batch size in single node
gradient_accumulation_steps=1 # requires more GPU memory
max_steps=-1
min_lr=0 # -> not used now
weight_decay=1e-4 # -> not used now
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false

accelerate launch --config_file=train/deepspeed_zero3.yaml \
    train/sft_qwen.py \
    --dataset_name ${dataset_name} \
    --model_name_or_path ${base_model} \
    --per_device_train_batch_size ${micro_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --output_dir training_outputs/sft-$(echo ${dataset_name} | rev | cut -d'-' -f1-5 | rev)-${base_model}-${uid} \
    --bf16 True \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --num_train_epochs ${epochs}  
    # --gradient_checkpointing=True  # Enable gradient checkpointing for efficient memory usage with 8 H100 GPUs.
    # --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'
