import torch
import torch.nn as nn
import pandas as pd
from torch.nn import Embedding
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
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

    for sample_batched in tqdm(dataloader):
        # get the inputs (batch)
        inputs, labels = sample_batched

        # 1 - zero the gradients
        optimizer.zero_grad()

        # 2 - forward pass: compute predicted y by passing x to the model
        outputs = model(inputs)
        # 3 - compute loss
        loss = loss_function(outputs.squeeze(), labels.float().squeeze())
        if ev_tr == "train":
            loss.backward()

            # 5 - update weights
            optimizer.step()
        loss_score.append(loss.detach().item())
    return np.average(loss_score)


# class SampleDataset(Dataset):
#  # in the case that i couldn't fit everything on memory i would use this code.
#     def __init__(self, path):
#         self.features = ['CountryPlayer', 'RoundCount', 'Turnover']
#         self.labels = ['GGR']
#         self.path = path
#
#     def __getitem__(self, id):
#         chunk = pd.read_csv(self.path, chunksize=100000)
#         ft = ""
#         lb = ""
#         for temp in chunk:
#             res = temp.loc[temp.index.isin([id]), :]
#             if res.shape[0]:
#                 ft = res.loc[:, self.features]
#                 lb = res.loc[:, self.labels]
#                 ft, lb = (res.index[0], torch.FloatTensor(ft.values)), torch.FloatTensor(lb.values)
#                 break
#         return ft, lb
#
#     def __len__(self):
#         return pd.read_csv(self.path).shape[0]

class SampleDataset(Dataset):

    def __init__(self, path):
            self.features = ['CountryPlayer', 'RoundCount', 'Turnover']
            self.labels = ['GGR']
            self.data = pd.read_csv(path)

    def __getitem__(self, item):
        ft, lb = self.data.loc[item, self.features], self.data.loc[item, self.labels]
        return (item, torch.FloatTensor(ft)), torch.FloatTensor(lb)

    def __len__(self):
        return self.data.shape[0]


class EmdeddingTrainer(nn.Module):

    def retrieve_embedding(self, item):
        search_emb = torch.tensor(item, dtype=torch.long)
        if torch.cuda.is_available():
            search_emb = search_emb.cuda()
        return self.embs(search_emb)

    def __init__(self, total_players, embedding_dim=50, kernel_sizes=(3, 4, 5), output_size=1):
        super(EmdeddingTrainer, self).__init__()
        self.embs = Embedding(total_players, embedding_dim=embedding_dim)
        self.embs.weight.requires_grad = True
        self.convs = nn.ModuleList([nn.Conv1d(1, embedding_dim+3, K) for K in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * embedding_dim+3*len(kernel_sizes), output_size)

    def forward(self, x):
        idx, x = x
        retrieved_emb = self.retrieve_embedding(idx)
        x = torch.cat([retrieved_emb.unsqueeze(1), x.unsqueeze(1)], axis=2)
        inputs = [torch.relu(conv(x)) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        inputs = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]  # [(N,Co), ...]*len(Ks)
        concated = torch.cat(inputs, 1)
        out = self.fc(concated)
        return out


def dnn_main():
    BATCH = 10000
    EPOCHS = 1
    lr = 0.01

    X_path = "../data/player_features.csv"

    model = EmdeddingTrainer(pd.read_csv(X_path).shape[0])
    criterion = torch.nn.L1Loss()

    train_dataset = SampleDataset(X_path)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=False, num_workers=4)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(parameters, lr=lr)

    train_losses = []


    for i in range(EPOCHS):
        print(f"epoch {i}")
        train_loss = train_epoch(i, train_loader, model, criterion, optimizer, "train")
        train_losses.append(train_loss)
        print(f"train loss {train_loss}")

    with open("../data/player_embeddings.pickle", "wb") as f:
        tmp = pd.DataFrame(model.embs.weight.detach())
        players = pd.read_csv(X_path)["Player_DWID"]
        tmp.index = players
        pickle.dump(tmp, f)
