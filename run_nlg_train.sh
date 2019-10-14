#export CUDA_VISIBLE_DEVICES=0,1

data_dir=/Users/gong/workspace/table2text-transformer/nlg_data
python train.py \
    --model_name 'nlg' \
    --train_files ${data_dir}/train.gtable ${data_dir}/train.gtable_label ${data_dir}/train.summary \
    --vocab_files ${data_dir}/train.gtable_vocab ${data_dir}/train.summary_vocab \
    --valid_files ${data_dir}/valid.gtable ${data_dir}/valid.summary \
    --num_encoder_layers 1 \
    --num_decoder_layers 2 \
    --share_source_target_embedding False \
    --share_embedding_and_softmax_weights True \
    --max_sequence_size 800 \
    --batch_size 4 \
    --constant_batch_size True \
    --lambda_cs 0.2 \
    --max_train_steps 100000 \
    --stopping_criterion 5 \
    --eval_metric 'nmt_bleu' \
    --save_periodic 200 \
    --eval_periodic 200 \
    --ngpus 0 \
    --decode_batch_size 4
