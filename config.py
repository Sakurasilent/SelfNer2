# 训练集路径
TRAIN_SAMPLE_PATH = './input/data/ResumeNER/train.char.bmes'
DEV_SAMPLE_PATH = './input/data/ResumeNER/dev.char.bmes'
# 标签路径
LABEL_PATH = './output/label.text'

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET_SIZE = 29
# 设小一点 意思一下
DROPOUT_PROB = 0.1
HIDDEN_SIZE = 512
