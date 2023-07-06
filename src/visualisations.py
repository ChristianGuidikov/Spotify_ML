import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import os.path

matplotlib.rcParams.update({'font.size': 8})
plt.rcParams['figure.dpi'] = 300

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "../data/spotify_data.csv")
df = pd.read_csv(open(path))


def plot_genre():
    """
        Graph:       1
        Dataframe:   genre_df
        Type:        Bar
        Description: plots the popularity of each genre of music as a result of the parameters given in dataframe
                     popularity_df
    """

    popularity_df = df[['artist_name', 'track_name', 'popularity', 'genre', 'acousticness']].query(
        "((acousticness >= 0.5 and genre=='acoustic') or genre!='acoustic') and popularity > 50").sort_values(
        by=['popularity'], ascending=False)

    genre_df = popularity_df[['genre']]
    genre_df['Counts'] = genre_df.groupby('genre')['genre'].transform('count')
    genre_df = genre_df.drop_duplicates().sort_values(by=['Counts'], ascending=False).iloc[0:10]

    # Uncomment the following to create a csv file of the dataframe in the results directory
    # filepath = Path('../results/visualisations/result1.csv')
    # genre_df.to_csv(filepath, index=False)

    x = np.asarray(genre_df.iloc[:, 0])
    y = np.asarray(genre_df.iloc[:, 1])
    ax = plt.bar(x, y, label='Popularity by genre')
    plt.xlabel('genre', size=15)  # define label for the horizontal axis
    plt.ylabel('popularity', size=15)  # define label for the vertical axis

    # Uncomment the following to create a png file of the plot
    # plt.savefig('../results/visualisations/result1.png')

    plt.show()


def plot_tempodance():
    """
        Graph:       2
        Dataframe:   tempodance_df
        Type:        Scatter
        Description: plots the correlation between tempo and dance-ability of songs
    """

    tempodance_df = df[['tempo', 'danceability']].iloc[0:1000000]

    # Uncomment the following to create a csv file of the dataframe in the results directory
    # filepath = Path('../results/visualisations/result2.csv')
    # tempodance_df.to_csv(filepath, index=False)

    x = np.asarray(tempodance_df.iloc[:, 0])
    y = np.asarray(tempodance_df.iloc[:, 1])
    ax = plt.scatter(x, y, label='Tempo in terms of dance-ability', s=1)
    plt.xlabel('tempo', size=15)
    plt.ylabel('dance-ability', size=15)

    # Uncomment the following to create a png file of the plot
    # plt.savefig('../results/visualisations/result2.png')

    plt.show()


def plot_genrelen():
    """
        Graph:       3
        Dataframe:   genrelen_df
        Type:        bar
        Description: plots the mean length of songs in genres (with acoustic having at least 50% confidence)
    """

    filter_df = df[['genre', 'acousticness', 'duration_ms']]\
        .query("(acousticness >= 0.5 and genre=='acoustic') or genre!='acoustic'")

    genrelen_df = filter_df[['genre', 'duration_ms']]
    genrelen_df['Mean'] = genrelen_df.groupby('genre')['duration_ms'].transform('mean').round()/60000
    genrelen_df = genrelen_df[['genre', 'Mean']]\
        .drop_duplicates()\
        .sort_values(by=['Mean'], ascending=False)\
        .iloc[0:30]

    # Uncomment the following to create a csv file of the dataframe in the results directory
    # filepath = Path('../results/visualisations/result3.csv')
    # genrelen_df.to_csv(filepath, index=False)

    x = np.asarray(genrelen_df.iloc[:, 0])
    y = np.asarray(genrelen_df.iloc[:, 1])
    matplotlib.rcParams.update({'font.size': 5})
    ax = plt.barh(x, y, label='Mean song length per genre')
    plt.gca().invert_yaxis()
    plt.xlabel('genre', size=15)  # define label for the horizontal axis
    plt.ylabel('mean', size=15)  # define label for the vertical axis

    # Uncomment the following to create a png file of the plot
    # plt.savefig('../results/visualisations/result3.png')

    plt.show()
