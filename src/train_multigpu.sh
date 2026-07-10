export NUM_NODES=1
export NUM_GPUS=2
accelerate launch --mixed_precision=bf16 --main_process_port 29505 --multi_gpu --num_machines $NUM_NODES --num_processes $NUM_GPUS train_difix.py \
    --output_dir=./outputs/difix/train_multi2 \
    --dataset_path="/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/data/data3.json" \
    --max_train_steps 15000 \
    --resolution=512 --learning_rate 2e-5 \
    --train_batch_size=4 --dataloader_num_workers 8 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=2000 --eval_freq 500 --viz_freq 10000 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
    --tracker_project_name "difix" --tracker_run_name "train" --timestep 199
