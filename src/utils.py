# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import sys
import pickle
import random
import getpass
import argparse
import subprocess
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from .logger import create_logger


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

MODEL_PATH = '/checkpoint/%s/dumped' % getpass.getuser()
DYNAMIC_COEFF = ['lambda_sm', 'lambda_mt', 'lambda_cs']


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    if params.local_rank == -1 or params.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not params.no_cuda else "cpu")
        params.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(params.local_rank)
        device = torch.device("cuda", params.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        params.n_gpu = 1

    params.device = device

    # dump parameters
    get_model_path(params)
    pickle.dump(params, open(os.path.join(params.model_path, 'params.pkl'), 'wb'))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0
    params.tensorboard_writer = SummaryWriter(log_dir=os.path.join(params.model_path, 'tensorboard'))

    # create a logger
    logger = create_logger(os.path.join(params.model_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.model_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   params.local_rank, params.device, params.n_gpu, bool(params.local_rank != -1))
    return logger


def get_model_path(params):
    """
    Create a directory to store the experiment.
    """
    model_path = MODEL_PATH if params.model_path == '' else params.model_path
    assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(model_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == '':
        chronos_job_id = os.environ.get('CHRONOS_JOB_ID')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            while True:
                exp_id = ''.join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.model_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.model_path):
        subprocess.Popen("mkdir -p %s" % params.model_path, shell=True).wait()


class AdamInverseSqrtWithWarmup(optim.Adam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
        decay_factor = lr * sqrt(warmup_updates)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        # linearly warmup for the first warmup_updates
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_updates ** 0.5
        for param_group in self.param_groups:
            param_group['num_updates'] = 0

    def get_lr_for_step(self, num_updates):
        # update learning rate
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * (num_updates ** -0.5)

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group['num_updates'] += 1
            param_group['lr'] = self.get_lr_for_step(param_group['num_updates'])


def get_optimizer(parameters, s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
    return AdamInverseSqrtWithWarmup(parameters, **optim_params)


def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    return [None if x is None else x.cuda() for x in args]


def restore_segmentation(path):
    """
    Take a file segmented with BPE and restore it to its original segmentation.
    """
    assert os.path.isfile(path)
    restore_cmd = "sed -i -r 's/(@@ )|(@@ ?$)//g' %s"
    subprocess.Popen(restore_cmd % path, shell=True).wait()


def parse_lambda_config(params):
    """
    Parse the configuration of lambda coefficient (for scheduling).
    x = "3"                  # lambda will be a constant equal to x
    x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease to 0 during the first 1000 iterations
    x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000 iterations, then will linearly increase to 1 until iteration 2000
    """
    for name in DYNAMIC_COEFF:
        if not hasattr(params, name):
            continue
        x = getattr(params, name)
        split = x.split(',')
        if len(split) == 1:
            setattr(params, name, float(x))
            setattr(params, name + '_config', None)
        else:
            split = [s.split(':') for s in split]
            assert all(len(s) == 2 for s in split)
            assert all(k.isdigit() for k, _ in split)
            assert all(int(split[i][0]) < int(split[i + 1][0]) for i in range(len(split) - 1))
            setattr(params, name, float(split[0][1]))
            setattr(params, name + '_config', [(int(k), float(v)) for k, v in split])


def get_lambda_value(config, n_iter):
    """
    Compute a lambda value according to its schedule configuration.
    """
    ranges = [i for i in range(len(config) - 1) if config[i][0] <= n_iter < config[i + 1][0]]
    if len(ranges) == 0:
        assert n_iter >= config[-1][0]
        return config[-1][1]
    assert len(ranges) == 1
    i = ranges[0]
    x_a, y_a = config[i]
    x_b, y_b = config[i + 1]
    return y_a + (n_iter - x_a) * float(y_b - y_a) / float(x_b - x_a)


def update_lambdas(params, n_iter):
    """
    Update all lambda coefficients.
    """
    for name in DYNAMIC_COEFF:
        if not hasattr(params, name):
            continue
        config = getattr(params, name + '_config')
        if config is not None:
            setattr(params, name, get_lambda_value(config, n_iter))


def set_sampling_probs(data, params):
    """
    Set the probability of sampling specific languages / language pairs during training.
    """
    coeff = params.lg_sampling_factor
    if coeff == -1:
        return
    assert coeff > 0

    # monolingual data
    params.mono_list = [k for k, v in data['mono_stream'].items() if 'train' in v]
    if len(params.mono_list) > 0:
        probs = np.array([1.0 * len(data['mono_stream'][lang]['train']) for lang in params.mono_list])
        probs /= probs.sum()
        probs = np.array([p ** coeff for p in probs])
        probs /= probs.sum()
        params.mono_probs = probs

    # parallel data
    params.para_list = [k for k, v in data['para'].items() if 'train' in v]
    if len(params.para_list) > 0:
        probs = np.array([1.0 * len(data['para'][(l1, l2)]['train']) for (l1, l2) in params.para_list])
        probs /= probs.sum()
        probs = np.array([p ** coeff for p in probs])
        probs /= probs.sum()
        params.para_probs = probs


def concat_batches(x1, len1, lang1_id, x2, len2, lang2_id, pad_idx, eos_idx, reset_positions):
    """
    Concat batches with different languages.
    """
    assert reset_positions is False or lang1_id != lang2_id
    lengths = len1 + len2
    if not reset_positions:
        lengths -= 1
    slen, bs = lengths.max().item(), lengths.size(0)

    x = x1.new(slen, bs).fill_(pad_idx)
    x[:len1.max().item()].copy_(x1)
    positions = torch.arange(slen)[:, None].repeat(1, bs).to(x1.device)
    langs = x1.new(slen, bs).fill_(lang1_id)

    for i in range(bs):
        l1 = len1[i] if reset_positions else len1[i] - 1
        x[l1:l1 + len2[i], i].copy_(x2[:len2[i], i])
        if reset_positions:
            positions[l1:, i] -= len1[i]
        langs[l1:, i] = lang2_id

    assert (x == eos_idx).long().sum().item() == (4 if reset_positions else 3) * bs

    return x, lengths, positions, langs


def truncate(x, lengths, max_len, eos_index):
    """
    Truncate long sentences.
    """
    if lengths.max().item() > max_len:
        x = x[:max_len].clone()
        lengths = lengths.clone()
        for i in range(len(lengths)):
            if lengths[i] > max_len:
                lengths[i] = max_len
                x[max_len - 1, i] = eos_index
    return x, lengths


def shuf_order(langs, params=None, n=5):
    """
    Randomize training order.
    """
    if len(langs) == 0:
        return []

    if params is None:
        return [langs[i] for i in np.random.permutation(len(langs))]

    # sample monolingual and parallel languages separately
    mono = [l1 for l1, l2 in langs if l2 is None]
    para = [(l1, l2) for l1, l2 in langs if l2 is not None]

    # uniform / weighted sampling
    if params.lg_sampling_factor == -1:
        p_mono = None
        p_para = None
    else:
        p_mono = np.array([params.mono_probs[params.mono_list.index(k)] for k in mono])
        p_para = np.array([params.para_probs[params.para_list.index(tuple(sorted(k)))] for k in para])
        p_mono = p_mono / p_mono.sum()
        p_para = p_para / p_para.sum()

    s_mono = [mono[i] for i in np.random.choice(len(mono), size=min(n, len(mono)), p=p_mono, replace=True)] if len(mono) > 0 else []
    s_para = [para[i] for i in np.random.choice(len(para), size=min(n, len(para)), p=p_para, replace=True)] if len(para) > 0 else []

    assert len(s_mono) + len(s_para) > 0
    return [(lang, None) for lang in s_mono] + s_para

def assign_parameter(module, ndarray):
    assert module.shape == ndarray.shape
    module.data = torch.from_numpy(ndarray)

def load_tf_weights_in_tnmt(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    :param model:
    :param tf_checkpoint_path:
    :return:
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
              "https://www.tensorflow.org/install/ for installation instructions.")
        raise

    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    tf_params = {}
    for name, shape in init_vars:
        if not name.startswith('transformer'):
            continue
        if name.endswith('Adam') or name.endswith("Adam_1"):
            continue
        # print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_params[name] = array

    # tf_params = pickle.load(open(model_file, 'rb'))
    # model_params = model.state_dict()

    encoder_ptr = getattr(model, 'encoder')
    decoder_ptr = getattr(model, 'decoder')
    encoder_embedding = getattr(encoder_ptr, 'embeddings')
    decoder_embedding = getattr(decoder_ptr, 'embeddings')

    assign_parameter(getattr(encoder_ptr, 'bias'), tf_params['transformer/bias'])
    assign_parameter(getattr(encoder_embedding, 'weight'), tf_params["transformer/source_embedding"])
    assign_parameter(getattr(decoder_embedding, 'weight'), tf_params["transformer/target_embedding"])

    for module_name in ['encoder', 'decoder']:
        module_ptr = getattr(model, module_name)
        for layer_id in range(6):
            # for module_name in ['attentions', 'layer_norm1', 'ffns', 'layer_norm2']:
            tf_prefix = "transformer/{}/layer_{}/self_attention/multihead_attention".format(module_name, layer_id)
            layer_ptr = getattr(getattr(module_ptr, 'attentions'), str(layer_id))

            qkv_matrix = tf_params[tf_prefix + "/qkv_transform/matrix"]
            q_lin_w = getattr(getattr(layer_ptr, 'q_lin'), 'weight')
            assign_parameter(q_lin_w, qkv_matrix[:, :512].transpose())
            k_lin_w = getattr(getattr(layer_ptr, 'k_lin'), 'weight')
            assign_parameter(k_lin_w, qkv_matrix[:, 512:1024].transpose())
            v_lin_w = getattr(getattr(layer_ptr, 'v_lin'), 'weight')
            assign_parameter(v_lin_w, qkv_matrix[:, 1024:1536].transpose())

            qkv_bias = tf_params[tf_prefix + "/qkv_transform/bias"]
            q_lin_b = getattr(getattr(layer_ptr, 'q_lin'), 'bias')
            assign_parameter(q_lin_b, qkv_bias[:512])
            k_lin_b = getattr(getattr(layer_ptr, 'k_lin'), 'bias')
            assign_parameter(k_lin_b, qkv_bias[512:1024])
            v_lin_b = getattr(getattr(layer_ptr, 'v_lin'), 'bias')
            assign_parameter(v_lin_b, qkv_bias[1024:1536])

            out_matrix = tf_params[tf_prefix + "/output_transform/matrix"]
            out_lin_w = getattr(getattr(layer_ptr, 'out_lin'), 'weight')
            assign_parameter(out_lin_w, out_matrix.transpose())

            out_bias = tf_params[tf_prefix + "/output_transform/bias"]
            out_lin_b = getattr(getattr(layer_ptr, 'out_lin'), 'bias')
            assign_parameter(out_lin_b, out_bias)

            tf_prefix = "transformer/{}/layer_{}/self_attention/layer_norm".format(module_name, layer_id)
            layer_ptr = getattr(getattr(module_ptr, 'layer_norm1'), str(layer_id))
            assign_parameter(getattr(layer_ptr, 'weight'), tf_params[tf_prefix + "/scale"])
            assign_parameter(getattr(layer_ptr, 'bias'), tf_params[tf_prefix + "/offset"])

            tf_prefix = "transformer/{}/layer_{}/feed_forward/ffn_layer".format(module_name, layer_id)
            layer_ptr = getattr(getattr(module_ptr, 'ffns'), str(layer_id))
            assign_parameter(getattr(getattr(layer_ptr, 'lin1'), 'weight'),
                             tf_params[tf_prefix + "/input_layer/linear/matrix"].transpose())
            assign_parameter(getattr(getattr(layer_ptr, 'lin1'), 'bias'),
                             tf_params[tf_prefix + "/input_layer/linear/bias"])
            assign_parameter(getattr(getattr(layer_ptr, 'lin2'), 'weight'),
                             tf_params[tf_prefix + "/output_layer/linear/matrix"].transpose())
            assign_parameter(getattr(getattr(layer_ptr, 'lin2'), 'bias'),
                             tf_params[tf_prefix + "/output_layer/linear/bias"])

            tf_prefix = "transformer/{}/layer_{}/feed_forward/layer_norm".format(module_name, layer_id)
            layer_ptr = getattr(getattr(module_ptr, 'layer_norm2'), str(layer_id))
            assign_parameter(getattr(layer_ptr, 'weight'), tf_params[tf_prefix + "/scale"])
            assign_parameter(getattr(layer_ptr, 'bias'), tf_params[tf_prefix + "/offset"])

            if module_name == 'decoder':
                tf_prefix = "transformer/{}/layer_{}/encdec_attention/multihead_attention".format(module_name,
                                                                                                  layer_id)
                layer_ptr = getattr(getattr(module_ptr, 'encoder_attn'), str(layer_id))

                q_matrix = tf_params[tf_prefix + "/q_transform/matrix"]
                q_lin_w = getattr(getattr(layer_ptr, 'q_lin'), 'weight')
                assign_parameter(q_lin_w, q_matrix.transpose())

                kv_matrix = tf_params[tf_prefix + "/kv_transform/matrix"]
                k_lin_w = getattr(getattr(layer_ptr, 'k_lin'), 'weight')
                assign_parameter(k_lin_w, kv_matrix[:, :512].transpose())
                v_lin_w = getattr(getattr(layer_ptr, 'v_lin'), 'weight')
                assign_parameter(v_lin_w, kv_matrix[:, 512:1024].transpose())

                q_bias = tf_params[tf_prefix + "/q_transform/bias"]
                q_lin_b = getattr(getattr(layer_ptr, 'q_lin'), 'bias')
                assign_parameter(q_lin_b, q_bias)

                kv_bias = tf_params[tf_prefix + "/kv_transform/bias"]
                k_lin_b = getattr(getattr(layer_ptr, 'k_lin'), 'bias')
                assign_parameter(k_lin_b, kv_bias[:512])
                v_lin_b = getattr(getattr(layer_ptr, 'v_lin'), 'bias')
                assign_parameter(v_lin_b, kv_bias[512:1024])

                out_matrix = tf_params[tf_prefix + "/output_transform/matrix"]
                out_lin_w = getattr(getattr(layer_ptr, 'out_lin'), 'weight')
                assign_parameter(out_lin_w, out_matrix.transpose())

                out_bias = tf_params[tf_prefix + "/output_transform/bias"]
                out_lin_b = getattr(getattr(layer_ptr, 'out_lin'), 'bias')
                assign_parameter(out_lin_b, out_bias)

                tf_prefix = "transformer/{}/layer_{}/encdec_attention/layer_norm".format(module_name, layer_id)
                layer_ptr = getattr(getattr(module_ptr, 'layer_norm15'), str(layer_id))
                assign_parameter(getattr(layer_ptr, 'weight'), tf_params[tf_prefix + "/scale"])
                assign_parameter(getattr(layer_ptr, 'bias'), tf_params[tf_prefix + "/offset"])
    return model
