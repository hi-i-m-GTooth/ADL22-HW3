for ((ii=0;ii<=50;ii+=5))
do
i=$(($ii-1))
accelerate launch infer_summary.py \
    --model_name_or_path "./results/50ep_warm/epoch_${i}/pytorch_model.bin" \
    --tokenizer_name "google/mt5-small" \
    --config_name "google/mt5-small" \
    --text_column "maintext" \
    --summary_column "title" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --max_length 512 \
    --seed 1231 \
    --test_file "_data/public.json" \
    --output "preds/${i}ep/sum_preds.jsonl" \
    --pad_to_max_length
    #--with_tracking

python eval.py -r data/public.jsonl -s preds/${i}ep/sum_preds.jsonl -d preds/${i}ep
done