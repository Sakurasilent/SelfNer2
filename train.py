import torch.utils.data

from config import *
from utils import *
from model import *

dataset = Dataset()
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     collate_fn=collate_fn
)

# model = BertCrfModel()
model = BertLSTMCrfModel()
model.to(device)
model.fine_tune(True)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
# criterion = torch.nn.CrossEntropyLoss()
for e in range(1):
    for b, (inputs, target) in enumerate(loader):
        mask = inputs['attention_mask'].bool()
        # input_ids = input['input_ids']
        y_pred = model(inputs, mask)
        loss = model.loss_fn(inputs, target, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if b % 10 == 0:
            print('epoch:', e, 'batch:', b, 'loss--->', loss)



