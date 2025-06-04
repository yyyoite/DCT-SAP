import sys
# HiSD的路径
sys.path.append('/xx/xx/HiSD/core')

from utils import get_config
from trainer import HiSD_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils

import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import time


def get_HiSD_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='/xx/HiSD/configs/celeba-hq_256.yaml')
    parser.add_argument('--checkpoint', type=str, default='/xx/HiSD/checkpoints/checkpoint_256_celeba-hq.pt')

    opts = parser.parse_args()

    config = get_config(opts.config)
    return opts, config


def get_HiSD_model(opts, config):
    trainer = HiSD_Trainer(config)
    state_dict = torch.load(opts.checkpoint)
    trainer.models.gen.load_state_dict(state_dict['gen_test'])
    trainer.models.gen.cuda()
    return trainer

def get_HiSD_img(trainer, x, steps, config):
    E = trainer.models.gen.encode
    T = trainer.models.gen.translate
    G = trainer.models.gen.decode
    M = trainer.models.gen.map
    F = trainer.models.gen.extract
    noise_dim = config['noise_dim']

    c = E(x)
    c_trg = c
    for j in range(len(steps)):
        step = steps[j]
        if step['type'] == 'latent-guided':
            if step['seed'] is not None:
                torch.manual_seed(step['seed'])
                torch.cuda.manual_seed(step['seed'])

            z = torch.randn(1, noise_dim).cuda()
            s_trg = M(z, step['tag'], step['attribute'])
        
        c_trg = T(c_trg, s_trg, step['tag'])
    x_trg = G(c_trg)
    return x_trg
