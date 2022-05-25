import torch
import torch.nn as nn
import pandas as pd
from torch.nn import Embedding
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
import numpy as np
torch.manual_seed(1)


def train_epoch(_epoch, dataloader, model, loss_function, optimizer, ev_tr):
    # switch to train mode -> enable regularization layers, such as Dropout
    if ev_tr == "train":
        model.train()
    elif ev_tr == "eval":
        model.eval()
    else:
        raise ValueError("NOT valid mode")
    loss_score = []

    for sample_batched in dataloader:
        # get the inputs (batch)
        inputs, labels = sample_batched

        # 1 - zero the gradients
        optimizer.zero_grad()

        # 2 - forward pass: compute predicted y by passing x to the model
        outputs = model(inputs)
        # 3 - compute loss
        loss = loss_function(outputs, labels.float())
        if ev_tr == "train":
            loss.backward()

            # 5 - update weights
            optimizer.step()

        loss_score.append(loss.detach().item())
    return np.average(loss_score)


class SampleDataset(Dataset):

    def __init__(self, path):
        features = ['CountryPlayer', 'RoundCount', 'Turnover']
        labels = ['GGR']
        pass

    def __getitem(self):
        pass

    def __len__(self):
        pass


class EmdeddingTrainer(nn.Module):

    def __init__(self, total_players, embedding_dim=50, kernel_sizes=(30, 40, 50), output_size=1):
        super(EmdeddingTrainer, self).__init__()
        self.embs = Embedding(len(total_players), embedding_dim=embedding_dim+3)
        self.embs.weight.requires_grad = True
        self.convs = nn.ModuleList([nn.Conv1d(1, embedding_dim+3, (K, 1)) for K in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * embedding_dim+3, output_size)

    def forward(self, x):
        inputs = x.unsqueeze(1).unsqueeze(-1)
        inputs = [torch.relu(conv(inputs)).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        inputs = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]  # [(N,Co), ...]*len(Ks)
        concated = torch.cat(inputs, 1)
        out = self.fc(concated)
        return out


BATCH = 10000
EPOCHS = 20
lr = 0.001

X_path = "../data/player_features.csv"

model = EmdeddingTrainer(pd.read_csv(X_path).shape[0])
criterion = torch.nn.L1Loss()

train_dataset = SampleDataset(X_path)
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0)
parameters = filter(lambda p: p.requires_grad, model.parameters())

optimizer = torch.optim.Adam(parameters, lr=lr)

train_losses = []
eval_losses = []

train_scores = []
eval_scores = []

for i in range(EPOCHS):
    train_loss = train_epoch(i, train_loader, model, criterion, optimizer, "train")
    train_losses.append(train_loss)

print("best evaluation scores are on {} {}".format(max(eval_scores), np.argmax(eval_scores)))
print("best model loss score is {} on {} epoch".format(min(eval_losses), np.argmin(eval_losses)))
