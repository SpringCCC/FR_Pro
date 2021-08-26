# @Time : 2021/8/26 23:10 
# @Author : WeiHuang
# @File : fpn.py 
# @Software: PyCharm
from torch import nn
from torch.nn import Conv2d
from utils.weight_init import normal_init




class FPN(nn.Module):

    def __init__(self, ):
        super(FPN, self).__init__()
        fpn_config = FPN_Config
        # in
        self.p1_in_net = Conv2d(fpn_config.p1_in_channel, fpn_config.out_channel, 1, 1, 0)
        self.p2_in_net = Conv2d(fpn_config.p2_in_channel, fpn_config.out_channel, 1, 1, 0)
        self.p3_in_net = Conv2d(fpn_config.p3_in_channel, fpn_config.out_channel, 1, 1, 0)
        # out
        self.p1_out_net = Conv2d(fpn_config.out_channel, fpn_config.out_channel, 3, 1, 1)
        self.p2_out_net = Conv2d(fpn_config.out_channel, fpn_config.out_channel, 3, 1, 1)
        self.p3_out_net = Conv2d(fpn_config.out_channel, fpn_config.out_channel, 3, 1, 1)
        # init
        normal_init(self.p1_in_net, 0, 0.001)
        normal_init(self.p2_in_net, 0, 0.001)
        normal_init(self.p3_in_net, 0, 0.001)
        normal_init(self.p1_out_net, 0, 0.001)
        normal_init(self.p2_out_net, 0, 0.001)
        normal_init(self.p3_out_net, 0, 0.001)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')




    def forward(self, p1, p2, p3):
        f1 = self.p1_in_net(p1)
        f2 = self.p1_in_net(p2)
        f3 = self.p1_in_net(p3)
        # top --> down
        # f3 --> f2 --> f1
        # p3 -- conv --f3 -------> f3
        #                   | upsample
        #                   f3_up
        #                   |
        # p2 -- conv --f2 --+------> f2
        #                   | upsample
        #                   f2_up
        #                   |
        # p1 -- conv --f1 --+------> f1
        f3_up = self.upsample(f3)
        f2 += f3_up
        f2_up = self.upsample(f2)
        f1 += f2_up
        return f1, f2, f3




class FPN_Config:
    # todo 待修改确认channel数目
    p1_in_channel = 128
    p2_in_channel = 128
    p3_in_channel = 128
    out_channel = 128

