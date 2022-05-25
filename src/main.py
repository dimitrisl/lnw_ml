import collections

import pymssql
import pandas as pd
import os
import torch
from torch.nn import Embedding


def clean_party(x):
    if "1st" in x.lower():
        x = "1st Party"
    elif "3rd" in x.lower():
        x = "3rd Party"
    return x

# Task 1 – Maximising revenue
# Via the casino powering platform players can play either games developed by other game providers (3rd party games)
# or games developed by SG Digital studios (1st party games). 1
# st party games are more profitable than 3rd party ones,
# so Dylan, the CEO would like to find a way to increase the percentage of players playing 1st party content, by
# recommending suitable games for each one of them.
# Create a Proof of Concept model that can identify the 3 most suitable 1st party games for a player, he hasn’t already
# tried. If you wish, you may try more than one approaches, and we can discuss over them during the 2nd interview.
# Please provide the necessary code for training your model and insights on its performance. You may prepare a short
# presentation to demonstrate your approach, results or any experiments you tried.
# You may use any Python library/framework of your choice.

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

sql_string = "select distinct  F.Game_DWID,IsSGDContent from factTablePlayer F inner join dimGame G on F.Game_DWID=G.Game_DWID INNER JOIN dimGameProvider P on P.GameProvider_DWID = G.GameProvider_DWID;"

gameid2party = pd.read_sql_query(sql_string, cnxn)
gameid2party["IsSGDContent"] = gameid2party["IsSGDContent"].apply(clean_party)
gameid2party.to_csv(f"../data/game2party.csv")
print("dumped gameid2party")
del gameid2party
sql_string = "select Player_DWID, SUM(GGR) as GGR,SUM(Turnover) as Turnover ,SUM(RoundCount) as RoundCount from FactTablePlayer GROUP BY Player_DWID;"

player_features = pd.read_sql_query(sql_string, cnxn)
player_features.to_csv("../data/player_features.csv")
print("dumped player_features")
del player_features
# i need to release resources
sql_string = "select Player_DWID, Game_DWID, count(BeginDate_DWID)  from FactTablePlayer GROUP BY Player_DWID, BeginDate_DWID, Game_DWID;"
# I decide to give a certain priority to the games each player likes on the final result signifying 0 to a game that he has never played before
# versus a game he has played more than 1 days.

player_likes = pd.read_sql_query(sql_string, cnxn)
player_likes.to_csv("../data/player_likes.csv")
print("dumped player_likes")
del player_likes
sql_string = "select distinct Player_DWID, Game_DWID from FactTablePlayer GROUP BY Player_DWID, Game_DWID;"
player_played = pd.read_sql_query(sql_string, cnxn)
player_played = player_played.groupby("Player_DWID")["Game_DWID"].agg(lambda x: list(x))
player_played.columns = ["Player_DWID", "Game_DWID"]
player_played.to_csv("../data/player_played.csv")
del player_played
print("dumped player_played")