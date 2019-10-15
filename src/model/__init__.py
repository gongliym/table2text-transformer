import torch
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

    if params.reload_model != "":
        logger.warning("Reloading model from %s ..." % params.reload_model)
        reloaded = torch.load(params.reload_model, map_location=torch.device('cpu'))['model']
        if all([k.startswith('module.') for k in reloaded.keys()]):
            reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}
        model.load_state_dict(reloaded)
    logger.debug("Model: {}".format(model))
    logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))
        
    return model

