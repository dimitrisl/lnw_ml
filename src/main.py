import numpy as np
import pymssql
import pandas as pd
import os
from player_clusterer import player_discovery
from dnn_models import dnn_main


def clean_party(x):
    if "1st" in x.lower():
        x = "1st Party"
    elif "3rd" in x.lower():
        x = "3rd Party"
    return x


def add_sgd(x, tmp):
    res = np.nan
    if x["Game_DWID"] in tmp:
        res = 1
    return res


def get_data():
    server = os.getenv("AS_SERVER")
    username = os.getenv("AS_USERNAME")
    database = os.getenv("AS_DATABASE")
    password = os.getenv("AS_PASSWORD")

    cnxn = pymssql.connect(server, username, password, database)
    # cursor = cnxn.cursor()

    table2columns = {
        "FactTablePlayer": ["BeginDate_DWID", "Player_DWID", "Operator_DWID", "Game_DWID", "CountryPlayer", "Turnover", "GGR", "RoundCount"],
        "dimOperator": ["Operator_DWID", "OperatorName"],
        "dimGame": ["Game_DWID", "GameName", "GameProvider_DWID"],
        "dimPlayer": ["Player_DWID", "playerid"],
        "dimGameProvider": ['GameProvider_DWID', 'GameProviderId', 'GameProviderName', 'IsSGDContent'],
    }

    sql_string = "select distinct  F.Player_DWID,F.Game_DWID,IsSGDContent from factTablePlayer F inner join dimGame G on F.Game_DWID=G.Game_DWID INNER JOIN dimGameProvider P on P.GameProvider_DWID = G.GameProvider_DWID where IsSGDContent like '%1st%';"

    gameid2party = pd.read_sql_query(sql_string, cnxn)
    gameid2party["IsSGDContent"] = gameid2party["IsSGDContent"].apply(clean_party)
    gameid2party.to_csv(f"../data/game2party.csv")
    print("dumped gameid2party")
    sql_string = "select Player_DWID, count(CountryPlayer) as CountryPlayer, SUM(GGR) as GGR,SUM(Turnover) as Turnover ,SUM(RoundCount) as RoundCount from FactTablePlayer GROUP BY Player_DWID;"

    tmp = gameid2party.loc[:, ["Game_DWID", "IsSGDContent"]].drop_duplicates()
    tmp.index = tmp.loc[:, "Game_DWID"]
    tmp = tmp.loc[:, ["IsSGDContent"]]
    tmp = tmp.to_dict()["IsSGDContent"]
    del gameid2party

    player_features = pd.read_sql_query(sql_string, cnxn)
    player_features.to_csv("../data/player_features.csv")
    print("dumped player_features")
    del player_features
    # i need to release resources
    sql_string = "select Player_DWID, Game_DWID, count(BeginDate_DWID) as play_times  from FactTablePlayer GROUP BY Player_DWID, BeginDate_DWID, Game_DWID;"
    # I decide to give a certain priority to the games each player likes on the final result signifying 0 to a game that he has never played before
    # versus a game he has played more than 1 days.

    player_likes = pd.read_sql_query(sql_string, cnxn)
    player_likes["isgd"] = player_likes.apply(lambda x: add_sgd(x, tmp), axis=1)
    del player_likes["isgd"]
    player_likes = player_likes.dropna()
    player_likes.to_csv("../data/player_likes.csv")
    total_players = sorted(player_likes.loc[:, "Player_DWID"].unique().tolist())
    print("dumped player_likes")
    del player_likes
    sql_string = "select distinct Player_DWID, Game_DWID from FactTablePlayer GROUP BY Player_DWID, Game_DWID;"
    player_played = pd.read_sql_query(sql_string, cnxn)
    player_played["isgd"] = player_played.apply(lambda x: add_sgd(x, tmp), axis=1)
    player_played = player_played.dropna()
    # we don't care to keep here the games that are not 1st party
    player_played = player_played.groupby("Player_DWID")["Game_DWID"].agg(lambda x: list(x))
    player_played.columns = ["Player_DWID", "Game_DWID"]
    player_played.to_csv("../data/player_played.csv")
    del player_played
    print("dumped player_played")


def central_pipeline():
    print("getting data")
    get_data()
    print("training embeddings")
    dnn_main()
    print("get neighbors of users")
    player_discovery()


central_pipeline()
