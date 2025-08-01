#!/bin/bash

# Arrays of parameter values to sweep
micro_batch_sizes=(2)
gradient_accumulation_steps=(32)
learning_rates=(1e-5 1e-6 1e-7)

# Base job name
base_job_name="sweep-QVL-7B-mc00"

# Counter for job submissions
job_count=0

echo "Starting PBS job sweep at $(date)"
echo "Micro batch sizes: ${micro_batch_sizes[@]}"
echo "Gradient accumulation steps: ${gradient_accumulation_steps[@]}"
echo "Learning rates: ${learning_rates[@]}"

# Create logs directory if it doesn't exist
mkdir -p pbs_queue_logs

# Usage: ./launch_sft_qwen_sweep_clean.sh
# Loop through all combinations
for mbs in "${micro_batch_sizes[@]}"; do
    for gas in "${gradient_accumulation_steps[@]}"; do
        for lr in "${learning_rates[@]}"; do
            # Create unique job name
            datetime=$(date +"%Y%m%d-%H%M%S")
            job_name="${base_job_name}-bs${mbs}-grad${gas}-lr${lr}-${datetime}"
            output_file="pbs_queue_logs/${job_name}.out"
            
            echo "Submitting job ${job_count}: ${job_name} (mbs=${mbs}, gas=${gas}, lr=${lr})"
            
            # Submit job with command line options and environment variables
            job_id=$(qsub -v MICRO_BATCH_SIZE="${mbs}",GRADIENT_ACCUMULATION_STEPS="${gas}",LEARNING_RATE="${lr}",PBS_OUTPUT_FILE="${output_file}" \
                    -N "${job_name}" -o "${output_file}" train/sft_qwen_parameterized.pbs)
            
            echo "  Job ID: ${job_id}"
            
            ((job_count++))
            
            # Optional: Add delay between submissions
            sleep 1
        done
    done
done

echo "Submitted ${job_count} jobs total"
echo "Use 'qstat -u $USER' to check job status" 