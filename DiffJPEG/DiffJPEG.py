# Pytorch
import torch
import torch.nn as nn
import torchvision
# Local
from .modules import compress_jpeg, decompress_jpeg
from .diffjpeg_utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor)

    def forward(self, x):
        '''

        '''
        x = (x + 1) / 2
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        recovered = recovered * 2 - 1
        return recovered

