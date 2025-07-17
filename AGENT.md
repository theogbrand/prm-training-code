# Agent Guidelines for PRM Training Code

## Build/Test Commands
- Run training: `bash train/sft_qwen.sh` or `bash train/sft_llama.sh` (main entry points)
- Setup environment: `source .venv/bin/activate` (alias: `sov`)
- Quick commit: `git add . && git commit -m "gfm"` (alias: `gfm`)
- GPU count check: `nvidia-smi -L | wc -l`

## Architecture & Structure
- **Main training scripts**: `train/` directory contains model-specific training scripts (qwen, llama, gemma, llava)
- **Data conversion**: `data_conversion/` for processing datasets (JSONL to Arrow/Parquet, HF uploads)
- **Models supported**: Qwen2.5-VL, Llama-3.2-Vision, Gemma-3, LLaVa, InternVL3
- **Training framework**: TRL (Transformers Reinforcement Learning) with HuggingFace
- **Distributed training**: Uses Accelerate + DeepSpeed Zero3 configuration
- **Storage**: Local datasets in `local_saved_datasets/`, outputs in `training_outputs/`

## Code Style
- **Imports**: Standard library first, then third-party (torch, transformers, trl), then local
- **Config**: Use dataclasses with field() for training arguments
- **Vision processing**: Qwen uses `qwen_vl_utils.process_vision_info()`, others use PIL Image format
- **Logging**: Use logging module with INFO level, include timestamps
- **Environment**: Load with `python-dotenv`, suppress FutureWarnings
- **Memory optimization**: Use bf16, gradient checkpointing, batch_size=1 with accumulation
