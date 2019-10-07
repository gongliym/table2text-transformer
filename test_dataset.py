from __future__ import absolute_import, division, print_function

import argparse
from src.utils import bool_flag
from src.data.data_loader import load_data, load_data2text_data

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Dataset Tester")

    ## data
    parser.add_argument("--train_files", nargs=3, type=str, required=True,
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
    parser.add_argument("--on_memory", type=bool_flag, default=True,
                        help="Load all data on memory.")

    return parser

def main(params):
    # load data
    train_data = load_data2text_data(params.train_files, params, train=False, repeat=False)
    for batch in train_data:
        print(batch)

if __name__ == "__main__":
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    main(params)
