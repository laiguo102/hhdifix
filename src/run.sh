#!/bin/bash
accelerate launch --mixed_precision=bf16 train_difix.py \
--output_dir=/home/bml/storage/mnt/v-zz4uoucip21b66el/Honghe/results/Difix3D_results_test \
--dataset_path="/home/bml/storage/mnt/v-zz4uoucip21b66el/Honghe/code/Difix3D_clean/data/data.json" \
--max_train_steps 20000 \
--resolution=512 --learning_rate 2e-5 \
--train_batch_size=4 --dataloader_num_workers 0 \
--enable_xformers_memory_efficient_attention \
--checkpointing_steps=5000 --eval_freq 5000 --viz_freq 5000 \
--lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
--tracker_project_name "difix" --tracker_run_name "train" --timestep 199 --mv_unet \

python inference_difix.py \
  --model_path "/home/bml/storage/mnt/v-zz4uoucip21b66el/Honghe/results/Difix3D_results_test/checkpoints/model_20001.pkl" \
  --input_image "/home/bml/storage/mnt/v-zz4uoucip21b66el/Honghe/data/metalens_0316/test/lr" \
  --ref_image "/home/bml/storage/mnt/v-zz4uoucip21b66el/Honghe/data/metalens_0316/test/uformer_3000real_260509" \
  --prompt "remove degradation" \
  --output_dir "/home/bml/storage/mnt/v-zz4uoucip21b66el/Honghe/data/metalens_0316/test/Difix3D_output_result" \
  --mv_unet

python evaluate_img.py -i /home/bml/storage/mnt/v-zz4uoucip21b66el/Honghe/data/metalens_0316/test/Difix3D_output_result -r /home/bml/storage/mnt/v-zz4uoucip21b66el/Honghe/data/metalens_0316/test/gt -o result.json