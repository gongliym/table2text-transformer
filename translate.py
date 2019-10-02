from __future__ import absolute_import, division, print_function

import argparse
from src.utils import bool_flag, initialize_exp, load_tf_weights_in_tnmt
from src.data.data_loader import load_data
from src.model import build_model
from src.data.vocab import Vocabulary
import torch

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
    parser.add_argument("--train_files",nargs=2, type=str, required=True,
                        help="Train data path")
    parser.add_argument("--vocab_files", nargs=2, type=str, required=True,
                        help="Vocabulary path")
    parser.add_argument("--input_file",type=str, required=True,
                        help="Input data path")
    parser.add_argument("--encoding", type=str, default='utf-8',
                        help="Data encoding (utf-8)")

    ## batch parameters
    parser.add_argument("--max_sequence_size", type=int, default=256,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--truncate_data", type=bool_flag, default=True,
                        help="truncate data when it's too long.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="batch size")
    parser.add_argument("--constant_batch_size", type=bool_flag, default=False,
                        help="use static batch size")
    parser.add_argument("--on_memory", type=bool_flag, default=False,
                        help="Load all data on memory.")

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

    # evaluation
    parser.add_argument("--beam_size", type=int, default=1,
                        help="beam size in beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="length penalty in beam search")
    parser.add_argument("--early_stopping", type=bool_flag, default=True,
                        help="early stopping in beam search")

    return parser

def load_test_data(input_file, vocab_file, params):
    vocab = Vocabulary(vocab_file)
    examples = []
    for line in open(input_file, 'r'):
        example = {}
        tokens = line.strip().split()
        example['source'] = [vocab[tok] for tok in tokens] + [vocab.eos_index]
        examples.append(example)

        if len(examples) >= params.batch_size:
            src_max_len = max(len(ex['source']) for ex in examples[:params.batch_size])
            src_padded, src_lengths = [], []
            for ex in examples[:params.batch_size]:
                src_seq = ex['source']
                src_padded.append(src_seq[:src_max_len] + [params.pad_index] * max(0, src_max_len - len(src_seq)))
                src_lengths.append(len(src_padded[-1]) - max(0, src_max_len - len(src_seq)))
            src_padded = torch.tensor(src_padded, dtype=torch.long)
            src_lengths = torch.tensor(src_lengths, dtype=torch.long)
            yield {'source': src_padded, 'source_length':src_lengths}
            examples = examples[params.batch_size:]
    if len(examples) > 0:
        src_max_len = max(len(ex['source']) for ex in examples)
        src_padded, src_lengths = [], []
        for ex in examples:
            src_seq = ex['source']
            src_padded.append(src_seq[:src_max_len] + [params.pad_index] * max(0, src_max_len - len(src_seq)))
            src_lengths.append(len(src_padded[-1]) - max(0, src_max_len - len(src_seq)))
        src_padded = torch.tensor(src_padded, dtype=torch.long)
        src_lengths = torch.tensor(src_lengths, dtype=torch.long)
        yield {'source': src_padded, 'source_length':src_lengths}

def main(params):
    initialize_exp(params)
    train_data = load_data(params.train_files, params, train=False, repeat=False)
    model = build_model(params)
    if params.tf_model_path != "":
        model = load_tf_weights_in_tnmt(model, params.tf_model_path)
    model.eval()

    for batch in load_test_data(params.input_file, params.vocab_files[0], params):
        output, out_len = model(mode='test',
                                src_seq=batch['source'],
                                src_len=batch['source_length'])

        for j in range(output.size(0)):
            sent = output[j,:]
            delimiters = (sent == params.eos_index).nonzero().view(-1)
            assert len(delimiters) >= 1 and delimiters[0].item() == 0
            sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]
            target = ' '.join([params.tgt_vocab.itos(sent[idx].item()) for idx in range(len(sent))])
            print(target)

if __name__ == "__main__":
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    main(params)
