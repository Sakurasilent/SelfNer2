import pandas as pd
from config import *


# 生成标签对
def generate_label():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[1], names=['label'], sep=' ')
    label_list = df['label'].value_counts().keys().to_list()
    label_dict = {v: k for k, v in enumerate(label_list)}
    label = pd.DataFrame(list(label_dict.items()))
    label.to_csv(LABEL_PATH, header=None, index=None)
    # return label_list


if __name__ == '__main__':
    generate_label()
