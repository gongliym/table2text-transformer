from __future__ import absolute_import, division, print_function

import argparse
from src.utils import bool_flag, initialize_exp, load_tf_weights_in_tnmt
from src.data.data_loader import load_data
from src.model import build_model
from src.trainer import EncDecTrainer

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Dataset Tester")

    # main parameters
    parser.add_argument("--model_path", type=str, default="./model_training/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="tmp",
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

    ## data
    parser.add_argument("--train_files", nargs=2, type=str, required=True,
                        help="Train data path")
    parser.add_argument("--vocab_files", nargs=2, type=str, required=True,
                        help="Vocabulary data path")
    parser.add_argument("--encoding", type=str, default='utf-8',
                        help="Data encoding (utf-8)")

    ## batch parameters
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
    parser.add_argument("--optimizer", type=str, default="adam_inverse_sqrt,lr=0.0007",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--lambda_mt", type=str, default="1",
                        help="MT coefficient")
    parser.add_argument("--max_train_epoches", type=int, default=20,
                        help="MT training epoches.")

    # reload pretrained embeddings / pretrained model / checkpoint
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # experiment parameters
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--no_cuda", type=bool_flag, default=False,
                        help="Avoid using CUDA when available")
    parser.add_argument("--multi_gpu", type=bool_flag, default=False,
                        help="using multiple gpus")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--is_master", type=bool_flag, default=False,
                        help="is master")
    return parser

def main(params):
    logger = initialize_exp(params)
    # load data
    train_data = load_data(params.train_files, params, train=False, repeat=False)
    model = build_model(params)
    if params.tf_model_path != "":
        model = load_tf_weights_in_tnmt(model, params.tf_model_path)

    total_num_parameters = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            logger.info("Trainable parameter: %s %s" % (name, parameter.size()))
            total_num_parameters += parameter.numel()

    logger.info("Total trainable parameter number: %d" % total_num_parameters)
    trainer = EncDecTrainer(model, train_data, params)

    trainer.checkpoint(None)
    while trainer.epoch <= params.max_train_epoches:
        trainer.mt_step(params.lambda_mt)
        trainer.iter()

if __name__ == "__main__":
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    main(params)
