import pandas as pd

from config import *
import torch
from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('hfl/rbt6')


# 获取 word与label对应list
def get_words_labels(path):
    words, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        words_item = []
        labels_item = []
        while True:
            cont = f.readline().replace('\n', '')
            if not cont:
                if not words_item and not labels_item:
                    # print("not")
                    break
                words.append(words_item)
                labels.append(labels_item)
                words_item = []
                labels_item = []
            # 不为空
            else:
                cont = cont.split(' ')

                words_item.append(cont[0])
                labels_item.append(cont[1])
    return [words, labels]


# 获取标签信息
def get_label():
    df = pd.read_csv(LABEL_PATH, names=['name', 'label'])
    # 返回label label2index index
    return list(df['label']), dict(df.values), list(df['label'])


# 构建Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        _, self.label2id, _ = get_label()
        # 先给一个 数据量少的默认值 dev
        self.dataset = get_words_labels(DEV_SAMPLE_PATH)
        print(' ')

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, item):
        tokens = self.dataset[0][item]
        labels = self.dataset[1][item]
        target = [self.label2id.get(l, self.label2id.get('O')) for l in labels]
        return tokens, target


def collate_fn(data):
    tokens = [i[0] for i in data]
    labels = [i[1] for i in data]
    _, label2id, _ = get_label()
    inputs = tokenizer.batch_encode_plus(tokens,

                                         padding=True,
                                         return_tensors='pt',
                                         is_split_into_words=True,

                                         )

    lens = inputs['input_ids'].shape[1]
    for i in range(len(labels)):
        labels[i] = [len(label2id)-1] + labels[i]
        labels[i] += [len(label2id)-1] * lens
        labels[i] = labels[i][:lens]
    return inputs.to(device), torch.LongTensor(labels).to(device)


if __name__ == '__main__':
    dataset = Dataset()
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=4,
                                         collate_fn=collate_fn,
                                         # shuffle=True,
                                         drop_last=True
                                         )
    # 查看数据样例
    for i, (inputs, labels) in enumerate(loader):
        break

    print(len(loader))
    print(tokenizer.decode(inputs['input_ids'][0]))
    print(labels[0])
    print(tokenizer.decode(inputs['input_ids'][1]))
    print(labels[1])
    print(tokenizer.decode(inputs['input_ids'][2]))
    print(labels[2])
    print(tokenizer.decode(inputs['input_ids'][3]))
    print(labels[3])
