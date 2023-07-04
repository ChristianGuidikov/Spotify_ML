import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({'font.size': 8})
plt.rcParams['figure.dpi'] = 300

df = pd.read_csv('./spotify_data.csv')
popularity_df = df[['artist_name', 'track_name', 'popularity', 'genre', 'acousticness']]\
    .query("((acousticness >= 0.5 and genre=='acoustic') or genre!='acoustic') and popularity > 50")\
    .sort_values(by=['popularity'], ascending=False)\


"""
    Graph:       1
    Dataframe:   genre_df
    Type:        Bar
    Description: plots the popularity of each genre of music as a result 
                 of the parameters given in dataframe popularity_df
"""
genre_df = popularity_df[['genre']]
genre_df['Counts'] = genre_df.groupby('genre')['genre'].transform('count')
genre_df = genre_df.drop_duplicates().sort_values(by=['Counts'], ascending=False).iloc[0:10]

X = np.asarray(genre_df.iloc[:, 0])
y = np.asarray(genre_df.iloc[:, 1])
ax = plt.bar(X, y, label='Popularity by genre')
plt.xlabel('popularity', size=15)  # define label for the horizontal axis
plt.ylabel('genre', size=15)  # define label for the vertical axis


# filepath = Path('./results/result2.csv')
# genre_df.to_csv(filepath)


plt.show()
