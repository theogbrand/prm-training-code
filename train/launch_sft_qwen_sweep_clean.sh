#!/bin/bash

# Arrays of parameter values to sweep
micro_batch_sizes=(1)
gradient_accumulation_steps=(2 4 8)

# Base job name
base_job_name="Qwen25-VL-32B-Instruct-mc00-lr2e-5-sweep"

# Counter for job submissions
job_count=0

echo "Starting PBS job sweep at $(date)"
echo "Micro batch sizes: ${micro_batch_sizes[@]}"
echo "Gradient accumulation steps: ${gradient_accumulation_steps[@]}"

# Create logs directory if it doesn't exist
mkdir -p pbs_queue_logs

# Usage: ./launch_sft_qwen_sweep_clean.sh
# Loop through all combinations
for mbs in "${micro_batch_sizes[@]}"; do
    for gas in "${gradient_accumulation_steps[@]}"; do
        # Create unique job name
        datetime=$(date +"%Y%m%d-%H%M%S")
        job_name="${base_job_name}-bs${mbs}-grad${gas}-${datetime}"
        output_file="pbs_queue_logs/${job_name}.out"
        
        echo "Submitting job ${job_count}: ${job_name} (mbs=${mbs}, gas=${gas})"
        
        # Submit job with command line options and environment variables
        job_id=$(qsub -v MICRO_BATCH_SIZE="${mbs}",GRADIENT_ACCUMULATION_STEPS="${gas}",PBS_OUTPUT_FILE="${output_file}" \
                -N "${job_name}" -o "${output_file}" train/sft_qwen_parameterized.pbs)
        
        echo "  Job ID: ${job_id}"
        
        ((job_count++))
        
        # Optional: Add delay between submissions
        sleep 1
    done
done

echo "Submitted ${job_count} jobs total"
echo "Use 'qstat -u $USER' to check job status" 