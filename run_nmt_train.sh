export CUDA_VISIBLE_DEVICES=0
python train_nmt.py \
	--train_files nmt_data/wmt2014.fr.BPE.shuf nmt_data/wmt2014.en.BPE.shuf \
	--vocab_files nmt_data/vocab_enfr nmt_data/vocab_enfr \
	--valid_files nmt_data/newstest2014.en-fr.norm.tok.tc.fr.BPE nmt_data/newstest2014.en-fr.norm.tok.tc.en \
	--share_source_target_embedding True \
	--eval_periodic 5000 \
	--batch_size 3600
