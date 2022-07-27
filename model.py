from premodel import *
import torch.nn as nn
from torchcrf import CRF
'''定义下游模型'''


class BertCrfModel(nn.Module):
    def __init__(self):
        super(BertCrfModel, self).__init__()
        self.pretrained = None
        # 默认不微调模型
        self.tuneing = False
        self.dropout = nn.Dropout(DROPOUT_PROB)
        self.classifier = nn.Linear(768, TARGET_SIZE)
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def forward(self, inputs, mask):
        out = self._get_bert_feature(inputs)
        return self.crf.decode(out, mask)

    def _get_bert_feature(self, inputs):
        # bert
        if self.tuneing:
            out = self.pretrained(**inputs).last_hidden_state
        else:
            with torch.no_grad():
                out = pretrain(**inputs).last_hidden_state
        return self.classifier(out)

    def loss_fn(self, inputs, target, mask):
        y_pred = self._get_bert_feature(inputs)
        return -self.crf.forward(y_pred, target, mask, reduction='mean')

    def fine_tune(self, tuneing):
        self.tuneing = tuneing
        if tuneing:
            for i in pretrain.parameters():
                i.requires_grd = True
            pretrain.train()
            self.pretrained = pretrain
        else:
            for i in pretrain.parameters():
                i.requires_grd(False)
            pretrain.eval()
            # 这里 研究 一下 怎么操作一下
            self.pretrained = None


if __name__ == '__main__':
    model = BertCrfModel()
    print(model)
