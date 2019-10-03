from logging import getLogger
from .transformer import Transformer

logger = getLogger()

def build_model(params):
    model = Transformer(params)

    total_num_parameters = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            logger.info("Trainable parameter: %s %s" % (name, parameter.size()))
            total_num_parameters += parameter.numel()
    logger.info("Total trainable parameter number: %d" % total_num_parameters)
    return model

