1. Pull dataset down from AWS, load to array, then dataset then train
    - rename base URL from S3 to current URL
    - make sure use list comprehension to process every item in the array. 
        - dataset.map() will convert image to byte array dict but we need it in PIL Image format for TRL SFTTrainer

2. edit S3 image path to local URL as per Qwen, then load into local dataset (don't pull from HF)

3. Models: llava-hf/llava-v1.6-mistral-7b-hf, meta-llama/Llama-3.2-11B-Vision-Instruct, Qwen/Qwen2.5-VL-7B-Instruct, google/gemma-3-12b-it, ?OpenGVLab/InternVL3-8B
    - Checked Llava/Llama no specific process_vision function 
    - Gemma uses: https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm_gemma3.py
    - Qwen use: https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L321