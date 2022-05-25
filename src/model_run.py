import pandas as pd
from torch.nn import Embedding
pi = pd.read_csv("../data/player_infos.csv").shape[0]


player_embeddings = Embedding(num_embeddings=pi, embedding_dim=300)




print('stop')

# table2columns["dimGameProvider"]["IsSGDContent"] = table2columns["dimGameProvider"]["IsSGDContent"].apply(clean_party)
#
# player_names = table2columns["dimPlayer"].Player_DWID.unique().tolist()
#
#
# pd.read_sql_query("select Player_DWID, SUM(GGR) as GGR,SUM(Turnover) as Turnover ,SUM(RoundCount) as RoundCount from FactTablePlayer GROUP BY Player_DWID;", cnxn).to_csv("../data/player_infos.csv",
#                                                                                                                         columns=["Player_DWID","GGR", "Turnover", "RoundCount"])
# pd.read_sql_query("select Game_DWID, SUM(GGR) as GGR,SUM(Turnover)  as Turnover ,SUM(RoundCount) as RoundCount from FactTablePlayer GROUP BY Game_DWID;", cnxn).to_csv("../data/item_infos.csv",
#                                                                                                                            columns=["Game_DWID","GGR", "Turnover", "RoundCount"])
#
#
# player_country = pd.read_sql_query("select distinct Player_DWID, CountryPlayer from FactTablePlayer;", cnxn)
