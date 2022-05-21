import pymssql
import pandas as pd
import os

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



# setted up credentials as enviromental variables
server = os.getenv("AS_SERVER")
username = os.getenv("AS_USERNAME")
database = os.getenv("AS_DATABASE")
password = os.getenv("AS_PASSWORD")

cnxn = pymssql.connect(server, username, password, database)
cursor = cnxn.cursor()

table2columns = {
    "FactTablePlayer": ",".join(["BeginDate_DWID", "Player_DWID", "Operator_DWID", "Game_DWID", "CountryPlayer", "Turnover", "GGR", "RoundCount"]),
    "dimOperator": ["ExternalOperatorID", "OperatorName"],
    "dimGame": ["GameID", "GameName", "GameProvider_DWID"],
    "dimPlayer": ["Player_DWID", "playerid"],
    "dimGameProvider": ["GameProviderID", "Game Provider Name", "IsSGDContent"],
}

lala = pd.read_sql(f"SELECT top 100 {table2columns['FactTablePlayer']} FROM FactTablePlayer ;", cnxn)  # this is gonna need a generator
# lala = pd.read_sql("SELECT * FROM dimGameProvider;", cnxn)  # this is gonna need a generator
# cursor.execute("SELECT * FROM FactTablePlayer;")
# for row in cursor:
#     print(row)
    # input("stop")
# cnxn.commit()
cnxn.close()

