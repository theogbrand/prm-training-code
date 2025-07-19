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
        
# def process_example_local(example):
#     """Load images from local files"""
#     pil_images = []
#     for s3_url in example['images']:
#         cwd_abs_path = os.path.abspath(os.getcwd())
#         local_path = s3_url.replace("s3://arf-share/arf-ob1-mm-reasoning/", cwd_abs_path + "/")
#         try:
#             if os.path.exists(local_path):
#                 pil_image = Image.open(local_path)
#                 pil_images.append(pil_image)  # Actually append the loaded image!
#             else:
#                 print(f"Warning: Local file not found: {local_path}")
#         except Exception as e:
#             print(f"Error loading {local_path}: {e}")
    
#     # Only update if we successfully loaded at least one image
#     if pil_images:
#         example['images'] = pil_images
#     else:
#         print("Warning: No images loaded for example")
#         example['images'] = []  # Keep it as empty list for consistency
    
#     return example

# def convert_sample_to_qwen_format(sample):
#     """
#     Convert sample from current schema to Qwen messages format
    
#     Args:
#         sample: Dict with "messages" and "images" keys
    
#     Returns:
#         Dict with "messages" key containing converted messages
#     """
#     messages = []
#     images = [sample.get("image", [])]
    
#     for message in sample["messages"]:
#         converted_message = {
#             "role": message["role"],
#             "content": []
#         }
        
#         for content_item in message["content"]:
#             if content_item["type"] == "text" and content_item["text"] is not None:
#                 converted_message["content"].append({
#                     "type": "text",
#                     "text": content_item["text"]
#                 })
#             elif content_item["type"] == "image":
#                 # Handle image by index
#                 image_index = content_item.get("index", 0)
#                 if images and image_index < len(images):
#                     converted_message["content"].append({
#                         "type": "image",
#                         "image": images[image_index]
#                     })
#                 else:
#                     # If no images provided, skip or handle as needed
#                     print(f"Warning: No image found for index {image_index}")
        
#         # Only add messages that have content
#         if converted_message["content"]:
#             messages.append(converted_message)
    
#     return {"messages": messages}

# def format_data_for_qwen(sample):
    # return {"messages": [
    #             {
    #                 "role": "system",
    #                 "content": [{"type": "text", "text": system_message}],
    #             },
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "text",
    #                         "text": prompt.format(product_name=sample["Product Name"], category=sample["Category"]),
    #                     },{
    #                         "type": "image",
    #                         "image": sample["image"],
    #                     }
    #                 ],
    #             },
    #             {
    #                 "role": "assistant",
    #                 "content": [{"type": "text", "text": sample["description"]}],
    #             },
    #         ],
    #     }

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
        training_args.run_name = f"{run_name_without_prefix}-{training_args.learning_rate}-{training_args.gradient_accumulation_steps}"
    else:
        training_args.run_name = f"{training_args.run_name}-{training_args.learning_rate}-{training_args.gradient_accumulation_steps}"
    
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
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        images = [[example["image"]] for example in examples]
        # if isinstance(model, LlavaForConditionalGeneration):
        #     # LLava1.5 does not support multiple images
        #     images = [image[0] for image in images]

        # Tokenize the texts and process the images
        # You can now use data_args.max_pixels and data_args.min_pixels here if needed
        # For example, when processing images with specific size constraints
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        processor.max_pixels = data_args.max_pixels
        processor.min_pixels = data_args.min_pixels

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
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

    # we cannot do this for large datasets, so we do it at dataset level
    # postprocessed_image_data = training_dataset["train"]
    # logging.info(f"example postprocessed_image_data[345]['messages']: {postprocessed_image_data[345]['messages']}")

    # postprocessed_image_data = [convert_sample_to_qwen_format(sample) for sample in training_dataset["train"]]
 

    # load dataset from JSONL file
   # Load your JSONL file
    # file_path = "/mnt/fast10/brandon/mmr_rollout_data/prm_training_data/train/AI2D_final_mc_rollouts_with_all_models_verification_merged_prm_training_data_final_trl_format_mc0.0.jsonl"

    # # Load data into a list
    # data = []
    # with open(file_path, 'r') as f:
    #     for line in f:
    #         data.append(json.loads(line.strip()))

    # print(f"Loaded {len(data)} samples") 
    # print(data[345]["messages"])

    # # need to use list comprehension to keep Pil.Image type, .map converts image to bytes
    # processed_data = [process_example_local(sample) for sample in data] 
    # postprocessed_image_data = [convert_sample_to_qwen_format(sample) for sample in processed_data] 
    
    # convert to HF Dataset for training
    # trainining_dataset = Dataset.from_list(postprocessed_image_data)  # type: ignore

    # assert isinstance(trainining_dataset[345]["images"][0], Image), "Image is not a PIL.Image.Image"

    # # Create dataset dict (optional, for train/validation split)
    # dataset_dict = DatasetDict({
    #     "train": training_dataset
    # })


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

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)