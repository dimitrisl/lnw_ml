import pickle
import faiss
import pandas as pd

with open("../data/player_embeddings.pickle", "rb") as f:
    embeddings = pickle.load(f)

embeddings["player_dwid"] = embeddings.index
embeddings.reset_index(drop=True, inplace=True)
index2player = embeddings.loc[:, "player_dwid"].to_dict()
embeddings = embeddings.loc[:, embeddings.columns != "player_dwid"]
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.values)


player_likes = pd.read_csv("../data/player_likes.csv")
print(" ")

D, I = index.search(embeddings.loc[0, :].values.reshape(1, -1), 10)
print("stop")
# this way we actually reduce our search space of users

# we cluster the user that are actually similar