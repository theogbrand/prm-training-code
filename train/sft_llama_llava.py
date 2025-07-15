import os
import logging
from datetime import datetime
from PIL import Image
import torch
from datasets import Dataset, DatasetDict, load_dataset
import json
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
pip install pillow

# Tested on 8x H100 GPUs
accelerate launch
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir sft-llava-1.5-7b-hf \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing

accelerate launch --config_file=train/deepspeed_zero3.yaml train/sft.py --dataset_name ob11/ai2d-prm-training-data-v0.1 --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct --per_device_train_batch_size 8 --gradient_accumulation_steps 8 --output_dir sft-meta-llama-3.2-11b-vision-instruct --bf16 True --torch_dtype bfloat16 --gradient_checkpointing

For LLaVA-NeXT, use: (requires transformers>=4.45)
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf

For meta-llama/Llama-3.2-11B-Vision-Instruct, use: (requires transformers>=4.45.1)
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct
"""
        
def process_example_local(example):
    """Load images from local files"""
    pil_images = []
    for s3_url in example['images']:
        cwd_abs_path = os.path.abspath(os.getcwd())
        local_path = s3_url.replace("s3://arf-share/arf-ob1-mm-reasoning/", cwd_abs_path + "/")
        try:
            if os.path.exists(local_path):
                pil_image = Image.open(local_path)
                pil_image_rgb = pil_image.convert("RGB")
                pil_images.append(pil_image_rgb)
                # pil_images.append(pil_image)  # Actually append the loaded image!
            else:
                logging.error(f"Warning: Local file not found: {local_path}")
        except Exception as e:
            logging.error(f"Error loading {local_path}: {e}")
    
    # Only update if we successfully loaded at least one image
    if pil_images:
        example['images'] = pil_images
    else:
        logging.error("Warning: No images loaded for example")
        example['images'] = []  # Keep it as empty list for consistency
    
    return example

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    
    # Set logging directory to output_dir with datetime suffix
    if training_args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_args.logging_dir = os.path.join(training_args.output_dir, "run_logs", f"run-{timestamp}")
        # Create the logging directory if it doesn't exist
        os.makedirs(training_args.logging_dir, exist_ok=True)
        logging.info(f"Logging directory set to: {training_args.logging_dir}")
    
    # Enable Weights & Biases reporting while keeping physical text logs
    training_args.report_to = ["wandb"]
    os.environ["WANDB_PROJECT"] = "multimodal-reasoning"
    os.environ["WANDB_ENTITY"] = "aisg-arf"
    logging.info("Enabled Weights & Biases reporting with project: multimodal-reasoning")
    
    # Set up file logging to the logging directory for physical text logs
    if training_args.logging_dir:
        log_file = os.path.join(training_args.logging_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Physical log file created at: {log_file}")

    logging.info("\n\nscript_args: %s", script_args)
    logging.info("\ntraining_args: %s", training_args)
    logging.info("\nmodel_args: %s", model_args)
    logging.info("\n\n")

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = None # full parameter
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    logging.info("\n\nmodel_kwargs: %s", model_kwargs)
    logging.info("\nprocessor: %s", processor)
    logging.info("\nmodel: %s", model)

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        images = [example["images"] for example in examples]
        if isinstance(model, LlavaForConditionalGeneration):
            # LLava1.5 does not support multiple images
            images = [image[0] for image in images]

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    ################
    # Dataset
    ################
    # training_dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # load dataset from JSONL file
   # Load your JSONL file
    file_path = "/home/ubuntu/porialab-us-south-2/ntu/brandon/prm-training-code/prm_training_data_full_v0/final_flattened_trl_format_prm_training_data_500k_mc0.8_v1.jsonl"

    # # Load data into a list
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    logging.info(f"loaded {len(data)} samples")
    logging.info(f"converting data to huggingface dataset {data[0]}")
    # convert to HF Dataset for training
    training_dataset = Dataset.from_list(data)  # type: ignore
    # need to use list comprehension to keep Pil.Image type, .map converts image to bytes
    logging.info(f"calling process_example_local on training_dataset, print sample: {training_dataset[0]}")
    processed_data = [process_example_local(sample) for sample in training_dataset]

    print("🔍 Verifying image types...")
    for i in range(min(3, len(processed_data))):
        img = processed_data[i]['images'][0]
        print(f"Sample {i}: {type(img)}")

    logging.info(f"converting processed_data to huggingface dataset {processed_data[0]}")
    processed_hf_dataset = Dataset.from_list(processed_data)  # type: ignore
    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=processed_hf_dataset, # train on full dataset for now
        eval_dataset=None,
        # eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        # peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)