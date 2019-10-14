#export CUDA_VISIBLE_DEVICES=0,1,2

data_dir=/Users/gong/workspace/table2text-transformer/nmt_data
python train.py \
    --model_name 'nmt' \
    --train_files ${data_dir}/sample.fr ${data_dir}/sample.en \
    --vocab_files ${data_dir}/vocab.fr ${data_dir}/vocab.en \
    --valid_files ${data_dir}/newstest2014.fr-en.fr.preprocessed.BPE ${data_dir}/newstest2014.fr-en.en.preprocessed \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --share_source_target_embedding False \
    --share_embedding_and_softmax_weights True \
    --max_sequence_size 256 \
    --batch_size 256 \
    --constant_batch_size False \
    --max_train_steps 100000 \
    --stopping_criterion 5 \
    --eval_metric 'nmt_bleu' \
    --save_periodic 200 \
    --eval_periodic 200 \
    --ngpus 0 \
    --decode_batch_size 10
