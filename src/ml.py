import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import sklearn

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "../data/spotify_data.csv")
df = pd.read_csv(open(path))


def plot_popgenres():
    """
        Application 1:  Which genres are going to be the most popular?
        Dataframe:      popgenres_df
        Model:          []
        Description:    Using model [] we were able to predict the trend of genre popularity
    """

    popgenres_df = df[['popularity', 'year', 'genre', 'danceability', 'acousticness']]\
        .query("((acousticness >= 0.5 and genre=='acoustic') or genre!='acoustic') and popularity>50")

    popgenres_df.drop(['danceability', 'acousticness'], axis=1, inplace=True)
    popgenres_df.sort_values(by=['year', 'genre'], inplace=True)

    popgenres_df['mean'] = popgenres_df.groupby(['year', 'genre'])['popularity'].transform('mean')
    popgenres_df.drop('popularity', axis=1, inplace=True)
    popgenres_df.drop_duplicates(inplace=True)
    popgenres_df.fillna(0, inplace=True)
    popgenres_df = popgenres_df[popgenres_df['year'] != 2023]

    # We can choose to represent all genres, but the plot will look crowded
    # genres = np.asarray(popgenres_df['genre'].drop_duplicates())

    # We choose the genres most popular based on the visualisations in part 1
    genres = ['hip-hop', 'dance', 'pop', 'alt-rock', 'country', 'indie-pop', 'k-pop', 'french']

    for genre in genres:
        x = np.asarray(popgenres_df.loc[popgenres_df['genre'] == genre]['year'])
        y = np.asarray(popgenres_df.loc[popgenres_df['genre'] == genre]['mean'])
        plt.plot(x, y, label=f'{genre}')

    plt.legend()

    # Uncomment the following to create a png file of the plot
    # plt.savefig('../results/ml/vis1.png')

    plt.show()


def plot_valence():
    """
        Application 2:  Are songs going to be more positive? (valence)
        Dataframe:      valence_df
        Model:          []
        Description:    Using model [] we were able to predict the trend of song positivity
    """

    valence_df = df[['year', 'genre', 'acousticness', 'valence']]\
        .query("(acousticness >= 0.5 and genre=='acoustic') or genre!='acoustic'")

    valence_df.drop('acousticness', axis=1, inplace=True)
    valence_df['mean_valence'] = valence_df.groupby(['year', 'genre'])['valence'].transform('mean')
    valence_df.drop('valence', axis=1, inplace=True)
    valence_df.drop_duplicates(inplace=True)
    valence_df.fillna(0, inplace=True)
    valence_df = valence_df[valence_df['year'] != 2023]
    valence_df.sort_values(by=['year', 'genre'], inplace=True)

    # We choose the genres most popular based on the visualisations in part 1
    genres = ['hip-hop', 'dance', 'pop', 'alt-rock', 'country', 'indie-pop', 'k-pop', 'french']

    for genre in genres:
        x = np.asarray(valence_df.loc[valence_df['genre'] == genre]['year'])
        y = np.asarray(valence_df.loc[valence_df['genre'] == genre]['mean_valence'])
        plt.plot(x, y, label=f'{genre}')

    plt.legend(fontsize="7")

    # Uncomment the following to create a png file of the plot
    # plt.savefig('../results/ml/vis2.png')

    plt.show()
