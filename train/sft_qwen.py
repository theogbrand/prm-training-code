import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
from PIL import Image
import torch
from datasets import Dataset, DatasetDict, load_dataset
import json
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
)
import dotenv
dotenv.load_dotenv()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DataArguments:
    """
    Arguments for data processing
    """
    max_pixels: int = field(
        default=256*28*28,  # Reduced from 576*28*28 for memory efficiency
        metadata={"help": "Maximum pixels for image processing (H*W)"}
    )
    min_pixels: int = field(
        default=16*28*28,
        metadata={"help": "Minimum pixels for image processing (H*W)"}
    )

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

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, DataArguments))
    script_args, training_args, model_args, data_args = parser.parse_args_and_config()
    
    # Memory-saving configurations to prevent OOM
    training_args.gradient_checkpointing = True
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    model_args.attn_implementation = "flash_attention_2"
    training_args.learning_rate = training_args.learning_rate * training_args.gradient_accumulation_steps # Linear scaling rule
    # training_args.learning_rate = training_args.learning_rate * (training_args.gradient_accumulation_steps ** 0.5) # square root scaling rule
    
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
    # Parse out training_outputs prefix and restructure run name
    if training_args.run_name.startswith("training_outputs/"):
        # Remove training_outputs/ prefix and keep the rest
        run_name_without_prefix = training_args.run_name.replace("training_outputs/", "", 1)
        training_args.run_name = f"{run_name_without_prefix}-lr-{training_args.learning_rate}-grad-steps-{training_args.gradient_accumulation_steps}-bs-{training_args.per_device_train_batch_size}"
    else:
        training_args.run_name = f"{training_args.run_name}-lr-{training_args.learning_rate}-grad-steps-{training_args.gradient_accumulation_steps}-bs-{training_args.per_device_train_batch_size}"
    
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
    logging.info("\ndata_args: %s", data_args)
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

    # Move model to GPU when using Flash Attention 2.0
    if model_args.attn_implementation == "flash_attention_2" and torch.cuda.is_available():
        model = model.to('cuda')
        logging.info("Model moved to GPU for Flash Attention 2.0")

    logging.info("\n\nmodel_kwargs: %s", model_kwargs)
    logging.info("\nprocessor: %s", processor)
    logging.info("\nmodel: %s", model)

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template with JIT cleaning
        texts = [
            processor.apply_chat_template(
                [
                    {
                        "role": message["role"],
                        "content": [
                            {"type": "image", "image": item["image"]}
                            for item in message["content"]
                            if item["type"] == "image" and item.get("image") is not None
                        ]
                        + [
                            {"type": "text", "text": item["text"]}
                            for item in message["content"]
                            if item["type"] == "text" and item.get("text") is not None
                        ],
                    }
                    for message in example["messages"]
                    if any(
                        (item["type"] == "text" and item.get("text") is not None)
                        or (item["type"] == "image" and item.get("image") is not None)
                        for item in message["content"]
                    )
                ],
                tokenize=False,
            )
            for example in examples
        ]
        # logging.info(f"CHECKING image token in texts: {texts[0]}")
        images = [[example["image"]] for example in examples]
        # logging.info(
        #     f"DEBUG: Images structure: {[type(img[0]) if img else 'None' for img in images]}"
        # )
        # if isinstance(model, LlavaForConditionalGeneration):
        #     # LLava1.5 does not support multiple images
        #     images = [image[0] for image in images]

        # Set processor constraints BEFORE processing
        processor.max_pixels = data_args.max_pixels
        processor.min_pixels = data_args.min_pixels

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652,151653,151655]
        else: 
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels

        return batch

    ################
    # Dataset
    ################
    if os.getenv("HF_TOKEN") is None:
        raise ValueError("HF_TOKEN is not set")
    else:
        logging.info(f"HF_TOKEN: {os.getenv('HF_TOKEN')[:5]}...")

    training_dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, token=os.getenv("HF_TOKEN"))
    
    # You can now use data_args.max_pixels and data_args.min_pixels in your dataset processing
    logging.info(f"Using max_pixels: {data_args.max_pixels}, min_pixels: {data_args.min_pixels}")

    # training_dataset = load_from_disk("prm-training-data-qwen")
    # logging.info(f"training_dataset: {training_dataset}")

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=training_dataset["train"], # train on full dataset for now
        eval_dataset=None,
        # eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        # peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)