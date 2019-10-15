export CUDA_VISIBLE_DEVICES=0,1
data_dir=/training/gong/table2text-transformer/nlg_data
python train.py \
    --model_name 'nlg' \
		--exp_id 'rl_finetuning' \
    --train_files ${data_dir}/train.gtable ${data_dir}/train.gtable_label ${data_dir}/train.summary \
    --vocab_files ${data_dir}/train.gtable_vocab ${data_dir}/train.summary_vocab \
    --valid_files ${data_dir}/valid.gtable ${data_dir}/valid.summary \
    --num_encoder_layers 1 \
    --num_decoder_layers 6 \
    --share_source_target_embedding False \
    --share_embedding_and_softmax_weights True \
    --max_sequence_size 700 \
    --batch_size 12 \
    --constant_batch_size True \
    --lambda_cs 0.2 \
    --max_train_steps 100000 \
    --stopping_criterion 5 \
    --eval_metric 'nmt_bleu' \
    --save_periodic 50 \
    --eval_periodic 50 \
    --ngpus 2 \
    --decode_batch_size 12 \
		--train_rl True \
		--lambda_rl 0.9 \
		--reload_model periodic-100000.pth \
		--optimizer "adam,lr=0.00001"
