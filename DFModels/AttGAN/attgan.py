import argparse
import json
import os
from os.path import join

import torch
import torch.utils.data as data
import torchvision.utils as vutils
import sys

from .model import AttGAN


def check_attribute_conflict(att_batch, att_name, att_names):
    def _get(att, att_name):
        if att_name in att_names:
            return att[att_names.index(att_name)]
        return None
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value
    att_id = att_names.index(att_name)
    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            if _get(att, 'Bangs') != 0:
                _set(att, 1-att[att_id], 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            for n in ['Bald', 'Receding_Hairline']:
                if _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
    return att_batch


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default='256_shortcut1_inject1_none_HQ')
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--num_test', dest='num_test', type=int)
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--custom_img', action='store_true')
    parser.add_argument('--custom_data', type=str, default='./data/custom')
    parser.add_argument('--custom_attr', type=str, default='./data/list_attr_custom.txt')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    return parser.parse_args(args)


def get_args():
    args_ = parse()
    # AttGAN的setting路径
    with open('/xx/xx/setting.txt', 'r') as f:
        args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

    args.test_int = args_.test_int
    args.num_test = args_.num_test
    # args.gpu = args_.gpu
    args.load_epoch = args_.load_epoch
    args.multi_gpu = args_.multi_gpu
    args.custom_img = args_.custom_img
    args.custom_data = args_.custom_data
    args.custom_attr = args_.custom_attr
    args.n_attrs = len(args.attrs)
    args.betas = (args.beta1, args.beta2)
    return args


def get_AttGAN_model(args):
    attgan = AttGAN(args)
    # AttGAN权重路径
    attgan.load('/xx/weights.199.pth')
    attgan.eval()
    return attgan


def get_AttGAN_img(attgan, x, att, args, index):
    att = att.type(torch.float)
    att_b_list = [att]
    for i in range(args.n_attrs):
        tmp = att.clone()
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        att_b_list.append(tmp)

    samples = [x]
    for i, att_b in enumerate(att_b_list):
        att_b_ = (att_b * 2 - 1) * args.thres_int
        if i > 0:
            att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
        samples.append(attgan.G(x, att_b_))
    samples = samples[2:]
    return samples[index]

def get_AttGAN_img_occluded(attgan, x, att, args, index):
    att = att.type(torch.float)
    batch_size = x.size(0)
    att = att.repeat(batch_size, 1)
    att_b_list = [att]
    
    for i in range(args.n_attrs):
        tmp = att.clone()
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        att_b_list.append(tmp)

    samples = [x]
    for i, att_b in enumerate(att_b_list):
        att_b_ = (att_b * 2 - 1) * args.thres_int
        if i > 0:
            att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
        samples.append(attgan.G(x, att_b_))
    samples = samples[2:]

    return samples[index]

