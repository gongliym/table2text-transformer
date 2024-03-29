from __future__ import absolute_import, division, print_function

import json
import argparse
import torch

from src.utils import bool_flag, initialize_exp, load_tf_weights_in_tnmt
from src.data.data_loader import load_data
from src.model import build_model
from src.trainer import EncDecTrainer
from src.evaluation.evaluator import TransformerEvaluator


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Train NMT")

    # main parameters
    parser.add_argument("--model_path", type=str, default="./model_training/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="nmt",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    parser.add_argument("--tf_model_path", type=str, default="",
                        help="Load from tensorflow model")

    # model parameters
    parser.add_argument("--model_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--hidden_dim", type=int, default=2048,
                        help="FFN inter-layer dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--num_encoder_layers", type=int, default=6,
                        help="Number of Transformer layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6,
                        help="Number of Transformer layers")
    parser.add_argument("--share_embedding_and_softmax_weights", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--share_source_target_embedding", type=bool_flag, default=False,
                        help="Share source and target embeddings.")

    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing")
    parser.add_argument("--attention_dropout", type=float, default=0.1,
                        help="Dropout in the attention layer")
    parser.add_argument("--residual_dropout", type=float, default=0.1,
                        help="Dropout")
    parser.add_argument("--relu_dropout", type=float, default=0.1,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")

    # data
    parser.add_argument("--train_files", nargs=2, type=str, required=True,
                        help="Train data path")
    parser.add_argument("--vocab_files", nargs=2, type=str, required=True,
                        help="Vocabulary data path")
    parser.add_argument("--valid_files", nargs=2, type=str, required=True,
                        help="Train data path")
    parser.add_argument("--encoding", type=str, default='utf-8',
                        help="Data encoding (utf-8)")

    # batch parameters
    parser.add_argument("--max_sequence_size", type=int, default=256,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--truncate_data", type=bool_flag, default=True,
                        help="truncate data when it's too long.")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="batch size")
    parser.add_argument("--constant_batch_size", type=bool_flag, default=False,
                        help="use static batch size")
    parser.add_argument("--on_memory", type=bool_flag, default=False,
                        help="Load all data on memory.")

    # training parameters
    parser.add_argument("--lambda_mt", type=str, default="1",
                        help="MT coefficient")
    parser.add_argument("--optimizer", type=str, default="adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0007",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--max_train_steps", type=int, default=1000000,
                        help="MT training epoches.")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")

    # reload pretrained embeddings / pretrained model / checkpoint
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # experiment parameters
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--eval_periodic", type=int, default=5000,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--no_cuda", type=bool_flag, default=False,
                        help="Avoid using CUDA when available")
    parser.add_argument("--multi_gpu", type=bool_flag, default=False,
                        help="using multiple gpus")
    parser.add_argument("--ngpus", type=int, default=0,
                        help="using multiple gpus")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--is_master", type=bool_flag, default=False,
                        help="is master")
    parser.add_argument("--only_eval", type=bool_flag, default=False,
                        help="only eval")

    # evaluation
    parser.add_argument("--beam_size", type=int, default=2,
                        help="beam size in beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="length penalty in beam search")
    parser.add_argument("--early_stopping", type=bool_flag, default=True,
                        help="early stopping in beam search")
    parser.add_argument("--decode_batch_size", type=int, default=4,
                        help="decode batch size")
    return parser


def main(params):
    logger = initialize_exp(params)
    model_name = 'transformer'
    # load data
    train_data = load_data(params.train_files, params, train=True, repeat=True, model=model_name)
    model = build_model(params, model=model_name)

    if params.tf_model_path != "":
        model = load_tf_weights_in_tnmt(model, params.tf_model_path)

    if params.ngpus >= 1:
        # model = torch.nn.DataParallel(model, device_ids=list(range(params.ngpus)))
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    trainer = EncDecTrainer(model, train_data, params)
    evaluator = TransformerEvaluator(trainer, params)

    if params.only_eval:
        scores = evaluator.run_all_evals(trainer.n_total_iter, model=model_name)
        # print / JSON log
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        return

    while trainer.n_total_iter <= params.max_train_steps:
        trainer.mt_step()
        trainer.iter()
        if params.eval_periodic > 0 and trainer.n_total_iter % params.eval_periodic == 0:
            # evaluate perplexity
            scores = evaluator.run_all_evals(trainer.n_total_iter, model=model_name)
            # print / JSON log
            for k, v in scores.items():
                logger.info("%s -> %.6f" % (k, v))
            logger.info("__log__:%s" % json.dumps(scores))
            params.tensorboard_writer.add_scalar('Evaluation/nmt_bleu', scores['nmt_bleu'], trainer.n_total_iter)

            # end of evaluation
            trainer.save_best_model(scores)
            trainer.save_periodic()
            trainer.end_evaluation(scores)


if __name__ == "__main__":
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    main(params)
