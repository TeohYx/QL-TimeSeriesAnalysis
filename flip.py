import pandas as pd

df = pd.read_csv('dataset/data_sm.csv', index_col=0)
print(df)
df = df.iloc[::-1]
print(df.index)

df.to_csv('dataset/data.csv')