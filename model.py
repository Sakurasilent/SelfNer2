from premodel import *
import torch.nn as nn

'''定义下游模型'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.pretrained = None
        # 默认不微调模型
        self.tuneing = False

    def forward(self, inputs):
        pass

    def fine_tune(self, tuneing):
        self.tuneing=tuneing
        pass







