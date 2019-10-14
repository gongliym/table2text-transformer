from logging import getLogger
from .transformer import Transformer
from .table2text_transformer import Data2TextTransformer

logger = getLogger()


def build_model(params, model="nmt"):
    if model == "nmt":
        model = Transformer(params)

        total_num_parameters = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                logger.info("Trainable parameter: %s %s" % (name, parameter.size()))
                total_num_parameters += parameter.numel()
        logger.info("Total trainable parameter number: %d" % total_num_parameters)
    elif model == "nlg":
        model = Data2TextTransformer(params)

        total_num_parameters = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                logger.info("Trainable parameter: %s %s" % (name, parameter.size()))
                total_num_parameters += parameter.numel()
        logger.info("Total trainable parameter number: %d" % total_num_parameters)
    else:
        raise Exception("Unkown model name. %s" % model)
    return model

