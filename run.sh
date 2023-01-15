python3.8 convert2jsonl.py -f $1

accelerate launch infer_summary.py \
    --model_name_or_path "./model_config/pytorch_model.bin" \
    --tokenizer_name "model_config" \
    --config_name "model_config" \
    --text_column "maintext" \
    --summary_column "title" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --max_length 512 \
    --seed 1231 \
    --test_file "_data/input.json" \
    --output $2 \
    --pad_to_max_length \
    --num_beams 4 --early_stopping