import pandas as pd
import numpy as np
from sklearn import preprocessing


'''
data = pd.read_csv('tags.csv')
data = data.drop(columns=['movieId','timestamp'])
data = data.applymap(str)
tags = data.groupby('userId')['tag'].agg(lambda tags: ','.join(tags)).reset_index()
userId = tags.ix[:,0]
userId = userId.astype(np.int64)
tags['userId'] = userId
tags = tags.sort_values(by='userId', ascending=True).reset_index(drop=True)
tags.to_csv('processed_tags.csv')
'''

movies = pd.read_csv('movies.csv')
movieIds = movies.ix[:,0]
print(movieIds)

ratings = pd.read_csv('ratings.csv')
userIds = ratings.userId.unique()
dfr = pd.DataFrame(0,index=userIds,columns=movieIds)

#ratings[ratings['userId']==1] 20 000263

print(ratings)
x = dfr[:100]
y = x.where(x<=0,1)
print(y.sum(axis=1))

for i in range(19000000,20000263):
    if i % 1000 == 0:
        print(i)
    u = ratings.iloc[i].userId
    m = ratings.iloc[i].movieId
    r = ratings.iloc[i].rating.astype(int)
    dfr.loc[u,m] = r


x = dfr[:100]
mask_dfr = x.where(x<=0,1)
count_dfr = mask_dfr.sum(axis=1)
print(count_dfr)
inds = small_dfr.loc[small_dfr>1000]
print(inds)
dfr.to_csv('processed_ratings.csv')

dfr = pd.read_csv('processed_ratings.csv')
