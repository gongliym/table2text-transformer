#export CUDA_VISIBLE_DEVICES=0,1
data_dir=/Users/gong/workspace/table2text-transformer/nlg_data/
python train.py \
    --model_name 'ie' \
    --train_files ${data_dir}/rl_train.examples \
    --vocab_files ${data_dir}/rl_train.vocab ${data_dir}/rl_train.label \
    --valid_files ${data_dir}//rl_train.examples \
    --max_sequence_size 128 \
    --batch_size 256 \
    --constant_batch_size False \
    --max_train_steps 100000 \
    --stopping_criterion 5 \
    --save_periodic 50 \
    --eval_periodic 50 \
    --decode_batch_size 12 \
		--optimizer "adam,lr=0.00001"
