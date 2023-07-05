import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({'font.size': 8})
plt.rcParams['figure.dpi'] = 300

df = pd.read_csv('./spotify_data.csv')


"""
    Graph:       1
    Dataframe:   genre_df
    Type:        Bar
    Description: plots the popularity of each genre of music as a result 
                 of the parameters given in dataframe popularity_df
"""
def plot_genre():
    popularity_df = df[['artist_name', 'track_name', 'popularity', 'genre', 'acousticness']].query(
        "((acousticness >= 0.5 and genre=='acoustic') or genre!='acoustic') and popularity > 50").sort_values(
        by=['popularity'], ascending=False)

    genre_df = popularity_df[['genre']]
    genre_df['Counts'] = genre_df.groupby('genre')['genre'].transform('count')
    genre_df = genre_df.drop_duplicates().sort_values(by=['Counts'], ascending=False).iloc[0:10]

    x = np.asarray(genre_df.iloc[:, 0])
    y = np.asarray(genre_df.iloc[:, 1])
    ax = plt.bar(x, y, label='Popularity by genre')
    plt.xlabel('popularity', size=15)  # define label for the horizontal axis
    plt.ylabel('genre', size=15)  # define label for the vertical axis
    plt.show()

    # Uncomment the following to create a csv file of the dataframe in the results directory
    # filepath = Path('./results/result1.csv')
    # genre_df.to_csv(filepath, index=False)


"""
    Graph:       2
    Dataframe:   tempodance_df
    Type:        Scatter
    Description: plots the correlation between tempo and dance-ability of songs
"""
def plot_tempodance():
    tempodance_df = df[['tempo', 'danceability']].iloc[0:1000000]

    x = np.asarray(tempodance_df.iloc[:, 0])
    y = np.asarray(tempodance_df.iloc[:, 1])
    ax = plt.scatter(x, y, label='Tempo in terms of dance-ability', s=1)
    plt.xlabel('tempo', size=15)
    plt.ylabel('dance-ability', size=15)
    plt.show()

    # Uncomment the following to create a csv file of the dataframe in the results directory
    # filepath = Path('./results/result2.csv')
    # tempodance_df.to_csv(filepath, index=False)


"""
    Graph:       3
    Dataframe:   genrelen_df
    Type:        bar
    Description: plots the mean length of songs in genres (with acoustic having at least 50% confidence)
"""
def plot_genrelen():
    filter_df = df[['genre', 'acousticness', 'duration_ms']]\
        .query("(acousticness >= 0.5 and genre=='acoustic') or genre!='acoustic'")

    genrelen_df = filter_df[['genre', 'duration_ms']]
    genrelen_df['Mean'] = genrelen_df.groupby('genre')['duration_ms'].transform('mean').round()/60000
    genrelen_df = genrelen_df[['genre', 'Mean']]\
        .drop_duplicates()\
        .sort_values(by=['Mean'], ascending=False)\
        .iloc[0:10]

    x = np.asarray(genrelen_df.iloc[:, 0])
    y = np.asarray(genrelen_df.iloc[:, 1])
    matplotlib.rcParams.update({'font.size': 5})
    ax = plt.bar(x, y, label='Mean song length per genre')
    plt.xlabel('genre', size=15)  # define label for the horizontal axis
    plt.ylabel('mean', size=15)  # define label for the vertical axis
    plt.show()

    # Uncomment the following to create a csv file of the dataframe in the results directory
    # filepath = Path('./results/result3.csv')
    # genrelen_df.to_csv(filepath, index=False)

