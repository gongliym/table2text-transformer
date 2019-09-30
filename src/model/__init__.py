from logging import getLogger
from .transformer import Transformer

logger = getLogger()

def build_model(params):
    model = Transformer(params)
    if params.device.type == 'cuda':
        return model.cuda()
    else:
        return model
