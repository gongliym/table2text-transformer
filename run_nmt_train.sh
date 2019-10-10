export CUDA_VISIBLE_DEVICES=0,1,2
python train_nmt.py \
    --train_files nmt_data/news_epps_un_common_giga.fr.BPE.shuf nmt_data/news_epps_un_common_giga.en.BPE.shuf \
    --vocab_files nmt_data/vocab.fr nmt_data/vocab.en \
    --valid_files evaluation/newstest2014.fr-en.fr.preprocessed.BPE evaluation/newstest2014.fr-en.en.preprocessed \
    --share_source_target_embedding False \
    --eval_periodic 5000 \
    --batch_size 12000 \
    --ngpus 3  \
    --decode_batch_size 60
