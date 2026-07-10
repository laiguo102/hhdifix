# accelerate launch --mixed_precision=bf16 train_difix.py \
#     --output_dir=./outputs/difix/train_multi_lora_pseudoswinir \
#     --dataset_path="/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/data/datapseudo.json" \
#     --max_train_steps 15000 \
#     --resolution=512 --learning_rate 2e-5 \
#     --train_batch_size=4 --dataloader_num_workers 0 \
#     --enable_xformers_memory_efficient_attention \
#     --checkpointing_steps=2000 --eval_freq 500 --viz_freq 10000 \
#     --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
#     --tracker_project_name "difix" --tracker_run_name "train" --timestep 199 --mv_unet

accelerate launch --mixed_precision=bf16 train_difix.py \
    --output_dir=./outputs/difix/train_lora_pseudonew \
    --dataset_path="/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/data/datapseudo.json" \
    --max_train_steps 15000 \
    --resolution=512 --learning_rate 2e-5 \
    --train_batch_size=4 --dataloader_num_workers 0 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=2000 --eval_freq 500 --viz_freq 10000 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
    --tracker_project_name "difix" --tracker_run_name "train" --timestep 199 --mv_unet