from __future__ import absolute_import, division, print_function

import os
import argparse
from src.utils import bool_flag, load_tf_weights_in_tnmt, AttrDict, to_cuda, restore_segmentation
from src.data.data_loader import load_data
from src.model import build_model
from src.data.table2text_dataset import load_and_batch_table_data
import torch


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="NMT translator")

    # main parameters
    parser.add_argument("--model_path", type=str,
                        help="Experiment dump path")
    parser.add_argument("--model_name", type=str, default="nmt",
                        help="model name")
    parser.add_argument("--tf_model_path", type=str, default="",
                        help="Load from tensorflow model")

    parser.add_argument("--input", type=str,
                        help="Experiment dump path")
    parser.add_argument("--output", type=str,
                        help="Experiment dump path")

    parser.add_argument("--decode_batch_size", type=int, default=1,
                        help="batch size")
    parser.add_argument("--beam_size", type=int, default=2,
                        help="beam size in beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="length penalty in beam search")
    parser.add_argument("--early_stopping", type=bool_flag, default=True,
                        help="early stopping in beam search")
    parser.add_argument("--no_cuda", type=bool_flag, default=True,
                        help="Avoid using CUDA when available")
    return parser


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() and not params.no_cuda else "cpu")
    reloaded = torch.load(params.model_path, map_location=device)
    model_params = AttrDict(reloaded['params'])
    model = build_model(model_params, model=params.model_name)
    model = torch.nn.DataParallel(model, device_ids=list(range(1)), dim=0)
    model.load_state_dict(reloaded['model'])
    if params.tf_model_path != "":
        model = load_tf_weights_in_tnmt(model, params.tf_model_path)
    setattr(model_params, "decode_batch_size", params.decode_batch_size)
    setattr(model_params, "beam_size", params.beam_size)
    setattr(model_params, "length_penalty", params.length_penalty)
    setattr(model_params, "early_stopping", params.early_stopping)

    if device.type == 'cuda':
        model = model.cuda()
    model.eval()

    outf = open(params.output, 'w', encoding='utf-8')
    for batch in load_and_batch_table_data(params.input, model_params):
        if device.type == 'cuda':
            for each in batch:
                batch[each] = batch[each].cuda()

        output = model(batch, mode='greedy')
        print(output)

        for j in range(output.size(0)):
            sent = output[j,:]
            delimiters = (sent == model_params.eos_index).nonzero().view(-1)
            assert len(delimiters) >= 1 and delimiters[0].item() == 0
            sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]
            target = ' '.join([model_params.tgt_vocab.itos(sent[idx].item()) for idx in range(len(sent))])
            print(target)
            outf.write(target+"\n")
        exit(0)
    outf.close()
    restore_segmentation(params.output)

if __name__ == "__main__":
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check model file
    assert os.path.isfile(params.model_path)

    with torch.no_grad():
        main(params)
