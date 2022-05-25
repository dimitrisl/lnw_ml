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
cursor = cnxn.cursor()

table2columns = {
    "FactTablePlayer": ["BeginDate_DWID", "Player_DWID", "Operator_DWID", "Game_DWID", "CountryPlayer", "Turnover", "GGR", "RoundCount"],
    "dimOperator": ["Operator_DWID", "OperatorName"],
    "dimGame": ["GameID", "GameName", "GameProvider_DWID"],
    "dimPlayer": ["Player_DWID", "playerid"],
    "dimGameProvider": ["GameProviderID", "GameProviderName", "IsSGDContent"],
}

for table in table2columns:
    print(table)
    tmp = ",".join(table2columns[table])
    sql_string = f"SELECT {tmp} from {table}"
    if table == "FactTablePlayer":
        dest = f"../data/{table}.csv"
        pd.DataFrame(columns=table2columns[table]).to_csv(dest)
        cx = pd.read_sql_query(sql_string, cnxn, chunksize=1000000)
        for i in cx:
            i.to_csv(f'../data/{table}.csv', index=False, mode='a', header=False)
    else:
        pd.read_sql_query(sql_string, cnxn).to_csv(f"../data/{table}.csv", columns=table2columns[table])

# table2columns["dimGameProvider"]["IsSGDContent"] = table2columns["dimGameProvider"]["IsSGDContent"].apply(clean_party)
#
# player_names = table2columns["dimPlayer"].Player_DWID.unique().tolist()
# player_embeddings = Embedding(num_embeddings=len(player_names), embedding_dim=300)
#
# pd.read_sql_query("select Player_DWID, SUM(GGR) as GGR,SUM(Turnover) as Turnover ,SUM(RoundCount) as RoundCount from FactTablePlayer GROUP BY Player_DWID;", cnxn).to_csv("../data/player_infos.csv",
#                                                                                                                         columns=["Player_DWID","GGR", "Turnover", "RoundCount"])
# pd.read_sql_query("select Game_DWID, SUM(GGR) as GGR,SUM(Turnover)  as Turnover ,SUM(RoundCount) as RoundCount from FactTablePlayer GROUP BY Game_DWID;", cnxn).to_csv("../data/item_infos.csv",
#                                                                                                                            columns=["Game_DWID","GGR", "Turnover", "RoundCount"])
#
#
# player_country = pd.read_sql_query("select distinct Player_DWID, CountryPlayer from FactTablePlayer;", cnxn)
