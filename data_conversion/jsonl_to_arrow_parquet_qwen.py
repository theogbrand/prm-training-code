import os
import json
import time
from datasets import DatasetDict, Dataset, Image
from PIL import Image as PILImage

input_jsonl_file = "/home/ubuntu/porialab-us-south-2/ntu/brandon/prm-training-code/prm_training_data_full_v0/final_flattened_trl_format_prm_training_data_500k_mc0.8_v1.jsonl"

def convert_images_to_qwen_format(sample):
    """
    Convert sample from current schema to Qwen messages format
    
    Args:
        sample: Dict with "messages" and "images" keys
    
    Returns:
        Dict with "messages" key containing converted messages
    """
    
    
    return {"messages": messages}

def process_example_local(example):
    """Load images from local files"""
    pil_images = []
    # print(f"example: {example}")
    for s3_url in example['images']:
        cwd_abs_path = os.path.abspath(os.getcwd())
        local_path = s3_url.replace("s3://arf-share/arf-ob1-mm-reasoning/", cwd_abs_path + "/")
        if os.path.exists(local_path):
            print(f"Appending image path: {local_path}")
            pil_images.append(local_path) # we cast to Image() later, here we just append the path
        else: 
            print(f"Warning: Local file not found: {local_path}")
            raise Exception(f"Local file not found: {local_path}")
    
    # Only update if we successfully loaded at least one image
    if pil_images:
        example['images'] = pil_images
    else:
        print("Warning: No images loaded for example")
        example['images'] = []  # Keep it as empty list for consistency

    return example['images']

messages_flat = []
images_flat = []
with open(input_jsonl_file, 'r', encoding='utf8') as f:
    for line in f:
        item = json.loads(line.strip())
        print(f"item: {item}")
        print(f"item['messages']: {item['messages']}")
        messages_flat.append(item['messages']) # just follow original format

        # 1. convert from S3 to local path for the ORIGINAL image.
        local_path_item = process_example_local(item)[0] # path in string format
        # print(f"local_path_item: {local_path_item}")

        # 2. Load and Resize the image by Qwen's default recommended size.


        # 3. save this updated image to a new local path, {original_image_path}_qwen_resized.png in the same directory
        # postprocessed_data = convert_images_to_qwen_format(local_path_item)
        # print(f"postprocessed_data: {postprocessed_data}")

        # 4. append this updated image path to the images_flat list, to be converted by Image() later
        
        
        images_flat.append(local_path_item['images'][0])  # Single path per row

print(f"images_flat[0]: {images_flat[0]}") # MUST BE A STRING ONLY NOT ARRAY
print(f"messages_flat[0]: {messages_flat[0]}") # Check in Qwen expected format
# create a Dataset instance from dict
hf_ds = Dataset.from_dict({"image": images_flat, "messages": messages_flat})

print(f"hf_ds after casing Dataset.from_dict: {hf_ds}")
for i in range(min(3, len(hf_ds))):
    img = hf_ds[i]['image']
    print(f"Sample {i}: {type(img)}")

print(f"now running cast column to Image() on hf_ds: {hf_ds}")
# cast the content of image column to PIL.Image
hf_ds = hf_ds.cast_column("image", Image())
# create train split
dataset = DatasetDict({"train": hf_ds})

training_dataset = dataset["train"]

for i in range(min(3, len(training_dataset))):
    img = training_dataset[i]['image']
    print(f"Sample {i}: {type(img)}")

# save Arrow files locally
# dataset.save_to_disk("cache")
# set num_proc to save faster with multiprocessing
dataset.save_to_disk("prm-training-data-qwen", num_proc=4)



# TODO (Later): for array of images instead of single images, explore using Sequence before casting to Image()
# from datasets import Features, Sequence, Image
# features = Features({"image": Sequence(Image()), "messages": ...})
# hf_ds = Dataset.from_dict({"image": images, "messages": messages}, features=features)