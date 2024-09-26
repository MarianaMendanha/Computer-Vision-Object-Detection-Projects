https://github.com/huggingface/transformers/blob/main/examples/pytorch/object-detection/README.md


python run_object_detection.py \
    --model_name_or_path facebook/detr-resnet-50 \
    --dataset_name cppe-5 \
    --do_train true \
    --do_eval true \
    --output_dir detr-finetuned-cppe-5-10k-steps \
    --num_train_epochs 100 \
    --image_square_size 600 \
    --fp16 true \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --remove_unused_columns false \
    --eval_do_concat_batches false \
    --ignore_mismatched_sizes true \
    --metric_for_best_model eval_map \
    --greater_is_better true \
    --load_best_model_at_end true \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --push_to_hub true \
    --push_to_hub_model_id detr-finetuned-cppe-5-10k-steps \
    --hub_strategy end \
    --seed 1337






python run_object_detection.py  --model_name_or_path MarianaMCruz/detr-finetuned-ppe --dataset_name cppe-5  --do_train true    --do_eval true    --output_dir detr-finetuned-ppe-retrained    --num_train_epochs 100    --image_square_size 600    --fp16 true   --learning_rate 2e-4    --weight_decay 1e-4     --dataloader_num_workers 4     --dataloader_prefetch_factor 2    --per_device_train_batch_size 8     --gradient_accumulation_steps 1     --remove_unused_columns false     --eval_do_concat_batches false    --ignore_mismatched_sizes true    --metric_for_best_model eval_map    --greater_is_better true    --load_best_model_at_end true    --logging_strategy epoch    --evaluation_strategy epoch    --save_strategy epoch    --save_total_limit 2    --push_to_hub true     --push_to_hub_model_id detr-finetuned-ppe-retrained     --hub_strategy end     --seed 1337 



python run_object_detection.py  --model_name_or_path MarianaMCruz/detr-finetuned-ppe   --do_train true    --do_eval true    --output_dir detr-finetuned-ppe-retrained    --num_train_epochs 100    --image_square_size 600    --fp16 true   --learning_rate 2e-4    --weight_decay 1e-4     --dataloader_num_workers 4     --dataloader_prefetch_factor 2    --per_device_train_batch_size 8     --gradient_accumulation_steps 1     --remove_unused_columns false     --eval_do_concat_batches false    --ignore_mismatched_sizes true    --metric_for_best_model eval_map    --greater_is_better true    --load_best_model_at_end true    --logging_strategy epoch    --evaluation_strategy epoch    --save_strategy epoch    --save_total_limit 2    --push_to_hub true     --push_to_hub_model_id detr-finetuned-ppe-retrained     --hub_strategy end     --seed 1337


python run_object_detection.py  --model_name_or_path facebook/detr-resnet-50   --dataset_name cppe-5   --do_train true    --do_eval true    --output_dir detr-finetuned-cppe-test    --num_train_epochs 100    --image_square_size 600    --fp16 true   --learning_rate 5e-5    --weight_decay 1e-4     --dataloader_num_workers 4     --dataloader_prefetch_factor 2    --per_device_train_batch_size 8     --gradient_accumulation_steps 1     --remove_unused_columns false     --eval_do_concat_batches false    --ignore_mismatched_sizes true    --metric_for_best_model eval_map    --greater_is_better true    --load_best_model_at_end true    --logging_strategy epoch    --evaluation_strategy epoch    --save_strategy epoch    --save_total_limit 2    --push_to_hub true     --push_to_hub_model_id detr-finetuned-cppe-test     --hub_strategy end     --seed 1337 --overwrite_output_dir


python run_object_detection.py  --model_name_or_path facebook/detr-resnet-50   --dataset_name Francesco/construction-safety-gsnvb   --do_train true    --do_eval true    --output_dir detr-finetuned-ppe    --num_train_epochs 100    --image_square_size 600    --fp16 true   --learning_rate 5e-5    --weight_decay 1e-4     --dataloader_num_workers 4     --dataloader_prefetch_factor 2    --per_device_train_batch_size 8     --gradient_accumulation_steps 1     --remove_unused_columns false     --eval_do_concat_batches false    --ignore_mismatched_sizes true    --metric_for_best_model eval_map    --greater_is_better true    --load_best_model_at_end true    --logging_strategy epoch    --evaluation_strategy epoch    --save_strategy epoch    --save_total_limit 2    --push_to_hub true     --push_to_hub_model_id detr-finetuned-ppe     --hub_strategy end     --seed 1337 --overwrite_output_dir



python run_object_detection.py  --model_name_or_path facebook/detr-resnet-50   --dataset_name cppe-5   --do_train true    --do_eval true    --output_dir detr-finetuned-cppe-test    --num_train_epochs 100    --image_square_size 600    --fp16 true   --learning_rate 5e-5    --weight_decay 1e-4     --dataloader_num_workers 4     --dataloader_prefetch_factor 2    --per_device_train_batch_size 8     --gradient_accumulation_steps 1     --remove_unused_columns false     --eval_do_concat_batches false    --ignore_mismatched_sizes true    --metric_for_best_model eval_map    --greater_is_better true    --load_best_model_at_end true    --logging_strategy epoch    --evaluation_strategy epoch    --save_strategy epoch    --save_total_limit 2    --push_to_hub true     --push_to_hub_model_id detr-finetuned-cppe-test     --hub_strategy end     --seed 1337



python run_object_detection.py  --model_name_or_path facebook/detr-resnet-50   --dataset_name cppe-5   --do_train true    --do_eval true    --output_dir detr-finetuned-cppe-5-10k-steps    --num_train_epochs 100    --image_square_size 600    --fp16 true   --learning_rate 5e-5    --weight_decay 1e-4     --dataloader_num_workers 4     --dataloader_prefetch_factor 2    --per_device_train_batch_size 8     --gradient_accumulation_steps 1     --remove_unused_columns false     --eval_do_concat_batches false    --ignore_mismatched_sizes true    --metric_for_best_model eval_map    --greater_is_better true    --load_best_model_at_end true    --logging_strategy epoch    --evaluation_strategy epoch    --save_strategy epoch    --save_total_limit 2    --push_to_hub true     --push_to_hub_model_id detr-finetuned-cppe-5-10k-steps     --hub_strategy end     --seed 1337


python run_object_detection.py     --model_name_or_path facebook/detr-resnet-50     --dataset_name Francesco/construction-safety-gsnvb     --do_train true    --do_eval true     --output_dir detr-finetuned-cppe-5-5epoch    --num_train_epochs 5     --image_square_size 600     --fp16 true     --learning_rate 5e-5     --weight_decay 1e-4     --dataloader_num_workers 2     --dataloader_prefetch_factor 2     --per_device_train_batch_size 8   --gradient_accumulation_steps 1     --remove_unused_columns false     --eval_do_concat_batches false     --ignore_mismatched_sizes true     --metric_for_best_model eval_map     --greater_is_better true     --load_best_model_at_end true     --logging_strategy epoch     --evaluation_strategy epoch    --save_strategy epoch    --save_total_limit 2    --push_to_hub true     --push_to_hub_model_id detr-finetuned-cppe-5-5epoch     --hub_strategy end     --seed 1337
    

python run_object_detection.py  --model_name_or_path facebook/detr-resnet-50  --do_train true    --do_eval true    --output_dir detr-finetuned-ppe    --num_train_epochs 100    --image_square_size 600    --fp16 true   --learning_rate 2e-4    --weight_decay 1e-4     --dataloader_num_workers 4     --dataloader_prefetch_factor 2    --per_device_train_batch_size 8     --gradient_accumulation_steps 1     --remove_unused_columns false     --eval_do_concat_batches false    --ignore_mismatched_sizes true    --metric_for_best_model eval_map    --greater_is_better true    --load_best_model_at_end true    --logging_strategy epoch    --evaluation_strategy epoch    --save_strategy epoch    --save_total_limit 2    --push_to_hub true     --push_to_hub_model_id detr-finetuned-ppe     --hub_strategy end     --seed 1337 --overwrite_output_dir


python run_object_detection.py  --model_name_or_path facebook/detr-resnet-50   --dataset_name keremberke/construction-safety-object-detection  --do_train true    --do_eval true    --output_dir detr-finetuned-ppe    --num_train_epochs 100    --image_square_size 600    --fp16 true   --learning_rate 2e-4    --weight_decay 1e-4     --dataloader_num_workers 4     --dataloader_prefetch_factor 2    --per_device_train_batch_size 8     --gradient_accumulation_steps 1     --remove_unused_columns false     --eval_do_concat_batches false    --ignore_mismatched_sizes true    --metric_for_best_model eval_map    --greater_is_better true    --load_best_model_at_end true    --logging_strategy epoch    --evaluation_strategy epoch    --save_strategy epoch    --save_total_limit 2    --push_to_hub true     --push_to_hub_model_id detr-finetuned-ppe     --hub_strategy end     --seed 1337


python run_object_detection.py  --model_name_or_path facebook/detr-resnet-50   --dataset_name Francesco/construction-safety-gsnvb   --do_train true    --do_eval true    --output_dir detr-finetuned-ppe    --num_train_epochs 100    --image_square_size 600    --fp16 true   --learning_rate 2e-4    --weight_decay 1e-4     --dataloader_num_workers 4     --dataloader_prefetch_factor 2    --per_device_train_batch_size 8     --gradient_accumulation_steps 1     --remove_unused_columns false     --eval_do_concat_batches false    --ignore_mismatched_sizes true    --metric_for_best_model eval_map    --greater_is_better true    --load_best_model_at_end true    --logging_strategy epoch    --evaluation_strategy epoch    --save_strategy epoch    --save_total_limit 2    --push_to_hub true     --push_to_hub_model_id detr-finetuned-ppe     --hub_strategy end     --seed 1337


python run_object_detection.py \    --model_name_or_path facebook/detr-resnet-50 \    --dataset_name cppe-5 \    --do_train true \    --do_eval true \    --output_dir detr-finetuned-cppe-5-10k-steps \    --num_train_epochs 100 \    --image_square_size 600 \    --fp16 true \    --learning_rate 5e-5 \    --weight_decay 1e-4 \    --dataloader_num_workers 4 \    --dataloader_prefetch_factor 2 \    --per_device_train_batch_size 8 \    --gradient_accumulation_steps 1 \    --remove_unused_columns false \    --eval_do_concat_batches false \    --ignore_mismatched_sizes true \    --metric_for_best_model eval_map \    --greater_is_better true \    --load_best_model_at_end true \    --logging_strategy epoch \    --evaluation_strategy epoch \    --save_strategy epoch \    --save_total_limit 2 \    --push_to_hub true \    --push_to_hub_model_id detr-finetuned-cppe-5-10k-steps \    --hub_strategy end \    --seed 1337

(.venv)










