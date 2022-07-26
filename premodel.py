from transformers import AutoModel
from config import *


pretrain = AutoModel.from_pretrained('hfl/rbt6')
pretrain.to(device)

if __name__ == '__main__':
    print(pretrain)