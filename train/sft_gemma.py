import io
import os
import zipfile
import logging
from datetime import datetime

import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
Train Gemma-3 on the HuggingFaceH4/llava-instruct-mix-vsft dataset (single-image).

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm_gemma3.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir gemma-3-4b-it-trl-sft-llava-instruct-mix-vsft \
    --bf16 \
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear \
    --attn_implementation eager

Train Gemma-3 on the FanqingM/MMIU-Benchmark dataset (multi-image).

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm_gemma3.py \
    --dataset_name FanqingM/MMIU-Benchmark \
    --dataset_train_split test \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir gemma-3-4b-it-trl-sft-MMIU-Benchmark \
    --bf16 \
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear
    --attn_implementation eager
"""
# same as Qwen version
def convert_sample_to_gemma_format(sample):
    """
    Convert sample from current schema to Qwen messages format
    
    Args:
        sample: Dict with "messages" and "images" keys
    
    Returns:
        Dict with "messages" key containing converted messages
    """
    messages = []
    images = sample.get("images", [])
    
    for message in sample["messages"]:
        converted_message = {
            "role": message["role"],
            "content": []
        }
        
        for content_item in message["content"]:
            if content_item["type"] == "text" and content_item["text"] is not None:
                converted_message["content"].append({
                    "type": "text",
                    "text": content_item["text"]
                })
            elif content_item["type"] == "image":
                # Handle image by index
                image_index = content_item.get("index", 0)
                if images and image_index < len(images):
                    converted_message["content"].append({
                        "type": "image",
                        "image": images[image_index]
                    })
                else:
                    # If no images provided, skip or handle as needed
                    print(f"Warning: No image found for index {image_index}")
        
        # Only add messages that have content
        if converted_message["content"]:
            messages.append(converted_message)
    
    return {"messages": messages}

# For multi-image example
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                if image is not None:
                    image = Image.open(io.BytesIO(image["bytes"]))
                    image_inputs.append(image.convert("RGB"))
    return image_inputs


# def format_data(samples: dict[str, any]) -> dict[str, list]:
#     formatted_samples = {"messages": []}
#     for cont in range(len(samples["question"])):
#         images = []
#         for img_path in samples["input_image_path"][cont]:
#             try:
#                 with open(img_path, "rb") as f:
#                     img_bytes = f.read()
#                 image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#                 images.append({"type": "image", "image": image})
#             except Exception as e:
#                 print(f"Error processing image {img_path}: {e}")
#                 continue

#         formatted_samples["messages"].append(
#             [
#                 {"role": "system", "content": [{"type": "text", "text": samples["context"][cont]}]},
#                 {"role": "user", "content": images + [{"type": "text", "text": samples["question"][cont]}]},
#                 {"role": "assistant", "content": [{"type": "text", "text": samples["output"][cont]}]},
#             ]
#         )
#     return formatted_samples


# # For multi-image example
# def prepare_dataset(dataset: DatasetDict, dataset_name: str, dataset_train_split: str) -> DatasetDict:
#     all_files = list_repo_files(dataset_name, repo_type="dataset")
#     zip_files = [f for f in all_files if f.endswith(".zip")]

#     for zip_filename in zip_files:
#         zip_path = hf_hub_download(repo_id=dataset_name, filename=zip_filename, repo_type="dataset")
#         extract_folder = zip_filename.replace(".zip", "")
#         os.makedirs(extract_folder, exist_ok=True)

#         with zipfile.ZipFile(zip_path, "r") as zip_ref:
#             zip_ref.extractall(extract_folder)

#     dataset = dataset.map(format_data, batched=True, batch_size=4, num_proc=16)
#     return dataset


def main():
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
    # quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    processor.tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    logging.info("\n\nmodel_kwargs: %s", model_kwargs)
    logging.info("\nprocessor: %s", processor)
    logging.info("\nmodel: %s", model)

    def collate_fn(examples):
        texts = [
            processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False).strip()
            for example in examples
        ]
        if "images" in examples[0]:  # single-image
            images = [[img.convert("RGB") for img in example["images"]] for example in examples]
        else:  # multi-image
            images = [process_vision_info(example["messages"]) for example in examples]

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        # Mask image tokens
        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["boi_token"])
        ]
        # Mask tokens for not being used in the loss computation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch  # Return the prepared batch

    ################
    # Dataset
    ################
    dataset = load_dataset("ob11/ai2d-prm-training-data-v0.4-pil", split="train")
    postprocessed_image_data = [convert_sample_to_gemma_format(sample) for sample in dataset]
    logging.info(f"example postprocessed_image_data[345]['messages']: {postprocessed_image_data[345]['messages']}")
    # if script_args.dataset_name == "FanqingM/MMIU-Benchmark":
    #     dataset = prepare_dataset(dataset, script_args.dataset_name, script_args.dataset_train_split)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=postprocessed_image_data,
        eval_dataset=None,
        # eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor,
        # peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    main()