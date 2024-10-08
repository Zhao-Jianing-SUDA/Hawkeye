

DATA_ROOT="dataset/vid_noaudio_split/train_new"
IMAGE_ROOT="dataset"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --master_port=29501 train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path dataset/new_train.json \
    --video_folder ${DATA_ROOT} \
    --image_folder ${IMAGE_ROOT} \
    --X "Video"\
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --pretrain_mm_mlp_adapter checkpoints/Video-LLaVA-Pretrain-7B/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/Video-LLaVA-7B \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir" \
    --output_dir "output_folder/Hawkeye"
