import pymssql
import pandas as pd
import os


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
    "FactTablePlayer": ",".join(["BeginDate_DWID", "Player_DWID", "Operator_DWID", "Game_DWID", "CountryPlayer", "Turnover", "GGR", "RoundCount"]),
    "dimOperator": ",".join(["Operator_DWID", "OperatorName"]),
    "dimGame": ",".join(["GameID", "GameName", "GameProvider_DWID"]),
    "dimPlayer": ",".join(["Player_DWID", "playerid"]),
    "dimGameProvider": ",".join(["GameProviderID", "GameProviderName", "IsSGDContent"]),
}

for table in table2columns:
    if table == "FactTablePlayer":
        continue
    sql_string = f"SELECT {table2columns[table]} from {table}"
    table2columns[table] = pd.read_sql_query(sql_string, cnxn)
    print(f"{table} shape {table2columns[table].shape}")

table2columns["dimGameProvider"]["IsSGDContent"] = table2columns["dimGameProvider"]["IsSGDContent"].apply(clean_party)

cursor.execute("SELECT  distinct Player_DWID FROM FactTablePlayer;")
lala_sql = """SELECT A.Player_DWID, B.Operator_DWID, B.Game_DWID FROM dimPlayer A
 INNER JOIN FactTablePlayer B ON A.Player_DWID = B.Player_DWID;"""

cnxn = pymssql.connect(server, username, password, database)
chunk = pd.read_sql_query(lala_sql, cnxn)
# get users and games on the same table
#,# chunksize=1000)
# for i in chunk:
#     print(i)

# this is gonna need a generator
# lala = pd.read_sql("SELECT * FROM dimGameProvider;", cnxn)  # this is gonna need a generator
# cursor.execute("SELECT * FROM FactTablePlayer;")
# for row in cursor:
#     print(row)
    # input("stop")
# cnxn.commit()
# cnxn.close()

