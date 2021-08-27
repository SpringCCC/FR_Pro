# -*- coding: utf-8 -*-
# @Time    : 2021/8/27 16:49
# @Author  : WeiHuang

import torch.nn as nn

class FPN_ProposalCreator(nn.Module):

    def __init__(self):
        super(FPN_ProposalCreator, self).__init__()
        