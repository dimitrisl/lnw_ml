import os
import pymssql
import pandas as pd

server = os.getenv("AS_SERVER")
username = os.getenv("AS_USERNAME")
database = os.getenv("AS_DATABASE")
password = os.getenv("AS_PASSWORD")

cnxn = pymssql.connect(server, username, password, database)


table2columns = {
    "FactTablePlayer": ["BeginDate_DWID", "Player_DWID", "Operator_DWID", "Game_DWID", "CountryPlayer", "Turnover", "GGR", "RoundCount"],
    "dimOperator": ["Operator_DWID", "OperatorName"],
    "dimGame": ["Game_DWID", "GameName", "GameProvider_DWID"],
    "dimPlayer": ["Player_DWID", "playerid"],
    "dimGameProvider": ['GameProvider_DWID', 'GameProviderId', 'GameProviderName', 'IsSGDContent'],
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


pd.read_sql_query("select Player_DWID, SUM(GGR) as GGR,SUM(Turnover) as Turnover ,SUM(RoundCount) as RoundCount from FactTablePlayer GROUP BY Player_DWID;", cnxn).to_csv("../data/player_infos.csv",
                  columns=["Player_DWID", "GGR", "Turnover", "RoundCount"])
print("player infos dumped")
pd.read_sql_query("select Game_DWID, SUM(GGR) as GGR,SUM(Turnover)  as Turnover ,SUM(RoundCount) as RoundCount from FactTablePlayer GROUP BY Game_DWID;", cnxn).to_csv("../data/item_infos.csv",
                  columns=["Game_DWID", "GGR", "Turnover", "RoundCount"])
print("games infos dumped")
