#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pytorch model inspector")
    parser.add_argument('--model', type=str, required=True, help = "the model file")
    parser.add_argument('--name', type=str, help = "parameter name")
    args = parser.parse_args()

    model = torch.load(args.model, map_location=torch.device('cpu'))['model']
    if args.name is None:
        for name in model.keys():
            print(name, model[name].size())
    else:
        print(model[args.name].size())
        print(model[args.name])
