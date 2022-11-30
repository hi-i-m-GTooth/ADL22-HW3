batch=32
# Greedy
accelerate launch infer_summary.py \
    --model_name_or_path "./results/50ep_warm/epoch_49/pytorch_model.bin" \
    --tokenizer_name "google/mt5-small" \
    --config_name "google/mt5-small" \
    --text_column "maintext" \
    --summary_column "title" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --max_length 512 \
    --seed 1231 \
    --test_file "_data/public.json" \
    --output "preds/49ep/sum_preds.jsonl" \
    --pad_to_max_length

python eval.py -r data/public.jsonl -s preds/49ep/sum_preds.jsonl -d preds/49ep

# Beam 10
accelerate launch infer_summary.py \
    --model_name_or_path "./results/50ep_warm/epoch_49/pytorch_model.bin" \
    --tokenizer_name "google/mt5-small" \
    --config_name "google/mt5-small" \
    --text_column "maintext" \
    --summary_column "title" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --max_length 512 \
    --seed 1231 \
    --test_file "_data/public.json" \
    --output "preds/49ep_beam10/sum_preds.jsonl" \
    --pad_to_max_length \
    --num_beams 10 --early_stopping

python eval.py -r data/public.jsonl -s preds/49ep_beam10/sum_preds.jsonl -d preds/49ep_beam10

# Beam 4
accelerate launch infer_summary.py \
    --model_name_or_path "./results/50ep_warm/epoch_49/pytorch_model.bin" \
    --tokenizer_name "google/mt5-small" \
    --config_name "google/mt5-small" \
    --text_column "maintext" \
    --summary_column "title" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --max_length 512 \
    --seed 1231 \
    --test_file "_data/public.json" \
    --output "preds/49ep_beam4/sum_preds.jsonl" \
    --pad_to_max_length \
    --num_beams 4 --early_stopping

python eval.py -r data/public.jsonl -s preds/49ep_beam4/sum_preds.jsonl -d preds/49ep_beam4

# Temp 0.7
accelerate launch infer_summary.py \
    --model_name_or_path "./results/50ep_warm/epoch_49/pytorch_model.bin" \
    --tokenizer_name "google/mt5-small" \
    --config_name "google/mt5-small" \
    --text_column "maintext" \
    --summary_column "title" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --max_length 512 \
    --seed 1231 \
    --test_file "_data/public.json" \
    --output "preds/49ep_temp07/sum_preds.jsonl" \
    --pad_to_max_length \
    --do_sample --top_k 0 --temperature 0.7

python eval.py -r data/public.jsonl -s preds/49ep_temp07/sum_preds.jsonl -d preds/49ep_temp07

# Temp 0.2
accelerate launch infer_summary.py \
    --model_name_or_path "./results/50ep_warm/epoch_49/pytorch_model.bin" \
    --tokenizer_name "google/mt5-small" \
    --config_name "google/mt5-small" \
    --text_column "maintext" \
    --summary_column "title" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --max_length 512 \
    --seed 1231 \
    --test_file "_data/public.json" \
    --output "preds/49ep_temp02/sum_preds.jsonl" \
    --pad_to_max_length \
    --do_sample --top_k 0 --temperature 0.2

python eval.py -r data/public.jsonl -s preds/49ep_temp02/sum_preds.jsonl -d preds/49ep_temp02

# Top-K10
accelerate launch infer_summary.py \
    --model_name_or_path "./results/50ep_warm/epoch_49/pytorch_model.bin" \
    --tokenizer_name "google/mt5-small" \
    --config_name "google/mt5-small" \
    --text_column "maintext" \
    --summary_column "title" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --max_length 512 \
    --seed 1231 \
    --test_file "_data/public.json" \
    --output "preds/49ep_k10/sum_preds.jsonl" \
    --pad_to_max_length \
    --do_sample --top_k 10

python eval.py -r data/public.jsonl -s preds/49ep_k10/sum_preds.jsonl -d preds/49ep_k10

# Top-K50
accelerate launch infer_summary.py \
    --model_name_or_path "./results/50ep_warm/epoch_49/pytorch_model.bin" \
    --tokenizer_name "google/mt5-small" \
    --config_name "google/mt5-small" \
    --text_column "maintext" \
    --summary_column "title" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --max_length 512 \
    --seed 1231 \
    --test_file "_data/public.json" \
    --output "preds/49ep_k50/sum_preds.jsonl" \
    --pad_to_max_length \
    --do_sample --top_k 50

python eval.py -r data/public.jsonl -s preds/49ep_k50/sum_preds.jsonl -d preds/49ep_k50

# Top-P094
accelerate launch infer_summary.py \
    --model_name_or_path "./results/50ep_warm/epoch_49/pytorch_model.bin" \
    --tokenizer_name "google/mt5-small" \
    --config_name "google/mt5-small" \
    --text_column "maintext" \
    --summary_column "title" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --max_length 512 \
    --seed 1231 \
    --test_file "_data/public.json" \
    --output "preds/49ep_p094/sum_preds.jsonl" \
    --pad_to_max_length \
    --do_sample --top_p 0.94

python eval.py -r data/public.jsonl -s preds/49ep_p094/sum_preds.jsonl -d preds/49ep_p094

# Top-P098
accelerate launch infer_summary.py \
    --model_name_or_path "./results/50ep_warm/epoch_49/pytorch_model.bin" \
    --tokenizer_name "google/mt5-small" \
    --config_name "google/mt5-small" \
    --text_column "maintext" \
    --summary_column "title" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --max_length 512 \
    --seed 1231 \
    --test_file "_data/public.json" \
    --output "preds/49ep_p098/sum_preds.jsonl" \
    --pad_to_max_length \
    --do_sample --top_p 0.98

python eval.py -r data/public.jsonl -s preds/49ep_p098/sum_preds.jsonl -d preds/49ep_p098