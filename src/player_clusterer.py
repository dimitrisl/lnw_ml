import pickle
import faiss
import pandas as pd


def player_discovery():
    with open("../data/player_embeddings.pickle", "rb") as f:
        embeddings = pickle.load(f)

    embeddings["player_dwid"] = embeddings.index
    player_likes_first = list(pd.read_csv("../data/player_played.csv").loc[:, "Player_DWID"].unique())
    emb2 = embeddings.loc[embeddings.index.isin(list(player_likes_first)), :]
    del emb2["player_dwid"]
    embeddings.reset_index(drop=True, inplace=True)
    index2player = embeddings.loc[:, "player_dwid"].to_dict()
    embeddings = embeddings.loc[:, embeddings.columns != "player_dwid"]
    index = faiss.IndexFlatL2(emb2.shape[1])
    index.add(emb2.values)

    pp = pd.read_csv("../data/player_played.csv")

    person_should_play = dict()
    for i in embeddings.index:
        temp_games = []
        D, I = index.search(embeddings.loc[i, :].values.reshape(1, -1), 10)
        D, I = D[:, 1:], I[:, 1:]
        player_id = index2player[i]
        try:
            games = str(pp.loc[pp["Player_DWID"] == player_id, "Game_DWID"].values[0]).strip('][').split(', ')
            games = [int(i) for i in games]
        except Exception:
            games = []
        for person in I.flatten():
            try:
                diff_games = str(pp.loc[pp["Player_DWID"] == index2player[person], "Game_DWID"].values[0]).strip('][').split(', ')
            except Exception:
                continue
            diff_games = [int(i) for i in diff_games]
            if not set(diff_games).intersection(games):
                diff_games.extend(temp_games)
            else:
                temp_games.extend(list(set(diff_games) - set(diff_games).intersection(games)))
            if len(temp_games) > 3:
                person_should_play[person] = temp_games
                break

    with open("../data/results.pickle", "wb") as f:
        pickle.dump(person_should_play, f)
