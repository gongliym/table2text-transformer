export CUDA_VISIBLE_DEVICES=0,1
python train_nlg.py \
    --train_files nlg_data/train.gtable nlg_data/train.gtable_label nlg_data/train.summary \
    --vocab_files nlg_data/train.gtable_vocab nlg_data/train.summary_vocab \
    --valid_files nlg_data/valid.gtable nlg_data/valid.summary \
    --num_encoder_layers 1 \
    --num_decoder_layers 6 \
    --max_sequence_size 800 \
    --batch_size 12 \
    --constant_batch_size True \
    --ngpus 2 \
    --eval_periodic 1000 \
    --decode_batch_size 12
