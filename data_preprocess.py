#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing

movies = pd.read_csv('movies.csv')
movieIds = movies.ix[:,0]
print(len(movieIds))
ratings = pd.read_csv('ratings.csv')
userIds = ratings.userId.unique()

y = ratings.groupby(['userId']).count()
x = ratings.groupby(['movieId']).count()
xinds = x.loc[x['userId'] >= 5000].index.to_numpy() #1005
yinds = y.loc[y['movieId'] >= 416].index.to_numpy() #10006

dfr = pd.DataFrame(0,index=yinds,columns=xinds)
dft = pd.DataFrame(0,index=yinds,columns=xinds)

filtered_ratings = ratings.loc[ratings['userId'].isin(yinds) & ratings['movieId'].isin(xinds)]
df = filtered_ratings.copy(deep=True)
print(df)

for i in range(len(df)):
    if i % 1000 == 0:
        print(i)
    u = df.iloc[i].userId
    m = df.iloc[i].movieId
    dfr.loc[u,m] = df.iloc[i].rating
    dft.loc[u,m] = df.iloc[i].timestamp

dft = dft.astype(int)
print(dft)
dft.to_csv('processed_times.csv')