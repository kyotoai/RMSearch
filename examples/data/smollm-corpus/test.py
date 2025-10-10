import json
import pandas as pd

df = pd.read_csv("df_small.csv")
df_small = df[:10]
df_small.to_csv("df_test.csv")