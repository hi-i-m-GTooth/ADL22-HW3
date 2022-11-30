accelerate launch summary.py \
    --model_name_or_path google/mt5-small \
    --text_column "maintext" \
    --summary_column "title" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 50 \
    --max_length 512 \
    --seed 1231 \
    --num_warmup_steps 20 \
    --checkpointing_steps "epoch" \
    --train_file "_data/train.json" \
    --validation_file "_data/public.json" \
    --output_dir "results/50ep_warm" \
    --pad_to_max_length
    #--with_tracking