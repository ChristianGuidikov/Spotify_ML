import datetime
import math

import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "../data/spotify_data.csv")
df = pd.read_csv(open(path))


def plot_popgenres():
    """
        Application 1:  Which genres are going to be the most popular?
        Dataframe:      popgenres_df
        Plot:           Plot of the most popular genres based on mean popularity over the years. We acknowledge here
                        that using a line plot instead of a scatter plot may be erroneous as it "creates" data points,
                        however it is easier to visualise.
        Model:          Polynomial Regression
        Description:    Using Polynomial Regression we were able to predict the trend of genre popularity amongst the
                        4 most popular genres.
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

    # We choose the most popular genres based on the visualisations in part 1
    genres = ['pop', 'hip-hop', 'dance', 'k-pop', 'alt-rock', 'country', 'indie-pop', 'french']

    for genre in genres:
        X = np.asarray(popgenres_df.loc[popgenres_df['genre'] == genre]['year'])
        y = np.asarray(popgenres_df.loc[popgenres_df['genre'] == genre]['mean'])
        plt.plot(X, y, label=f'{genre}')

    plt.legend()

    # Uncomment the following to create a png file of the plot
    # plt.savefig('../results/ml/vis1.png')

    plt.show()

    popgenres = ['pop', 'hip-hop', 'dance', 'k-pop']

    for genre in popgenres:
        X = np.asarray(popgenres_df.loc[popgenres_df['genre'] == genre]['year'])
        y = np.asarray(popgenres_df.loc[popgenres_df['genre'] == genre]['mean'])

        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(X.reshape(-1, 1))
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(poly_features, y)
        y_predicted = poly_reg_model.predict(poly_features)

        plt.scatter(X, y, label=f"{genre}", s=2)
        plt.plot(X, y_predicted, label=f"{genre} pred")

    plt.legend()

    # Uncomment the following to create a png file of the plot
    # plt.savefig('../results/ml/pred1.png')

    plt.show()


def plot_valence():
    """
        Application 2:  Are songs going to be more positive? (valence)
        Dataframe:      valence_df
        Plot:           Plot of the most popular genres and their valence (positivity).
        Model:          Polynomial Regression (3rd degree)
        Description:    Using model Polynomial Regression we were able to predict the trend of song positivity.
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
    genres = ['hip-hop', 'dance', 'pop', 'k-pop', 'alt-rock', 'country', 'indie-pop', 'french']

    for genre in genres:
        x = np.asarray(valence_df.loc[valence_df['genre'] == genre]['year'])
        y = np.asarray(valence_df.loc[valence_df['genre'] == genre]['mean_valence'])
        plt.plot(x, y, label=f'{genre}')

    plt.legend(fontsize="7")

    # Uncomment the following to create a png file of the plot
    # plt.savefig('../results/ml/vis2.png')

    plt.show()

    popgenres = ['hip-hop', 'dance', 'pop', 'k-pop']

    for genre in popgenres:
        X = np.asarray(valence_df.loc[valence_df['genre'] == genre]['year'])
        y = np.asarray(valence_df.loc[valence_df['genre'] == genre]['mean_valence'])

        poly = PolynomialFeatures(degree=3, include_bias=False)
        poly_features = poly.fit_transform(X.reshape(-1, 1))
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(poly_features, y)
        y_predicted = poly_reg_model.predict(poly_features)

        plt.scatter(X, y, label=f"{genre}", s=2)
        plt.plot(X, y_predicted, label=f"{genre} pred")

    plt.legend()

    # Uncomment the following to create a png file of the plot
    # plt.savefig('../results/ml/pred2.png')

    plt.show()


def plot_short():
    """
        Application 3:  Are songs getting shorter?
        Dataframe:      short_df
        Model:          Polynomial Regression
        Description:    Using model Polynomial Regression we were able to predict the trend of song duration over time.
    """

    short_df = df[['year', 'duration_ms']].query("year != 2023")
    short_df['mean_duration'] = short_df.groupby('year')['duration_ms'].transform('mean')/60000
    short_df.drop('duration_ms', axis=1, inplace=True)
    short_df.drop_duplicates(inplace=True)
    short_df.sort_values(by='year', inplace=True)

    X = np.asarray(short_df['year'])
    y = np.asarray(short_df['mean_duration'])

    fig, ax = plt.subplots()
    fig.canvas.draw()

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)
    y_predicted = poly_reg_model.predict(poly_features)

    ax.scatter(X, y, label="data")
    ax.plot(X, y_predicted, label="prediction", c="red")

    labels = ax.get_yticks().tolist()

    for i in range(len(labels)):
        labels[i] = round(labels[i], 1)
        labels[i] = f"${math.floor(labels[i])}:${round((labels[i]%1)*60)}"

    ax.set_yticklabels(labels)

    plt.legend()

    # Uncomment the following to create a png file of the plot
    # plt.savefig('../results/ml/pred3.png')

    plt.show()


def plot_tempo():
    """
        Application 4:  Are songs getting faster?
        Dataframe:      tempo_df
        Model:          []
        Description:    Using model [] we were able to predict the trend of tempo in music
    """

    tempo_df = df[['year', 'tempo']].query("year != 2023")
    tempo_df['mean_tempo'] = tempo_df.groupby('year')['tempo'].transform('mean')
    tempo_df.drop('tempo', axis=1, inplace=True)
    tempo_df.drop_duplicates(inplace=True)
    tempo_df.sort_values(by='year', inplace=True)

    X = np.asarray(tempo_df['year'])
    y = np.asarray(tempo_df['mean_tempo'])

    poly = PolynomialFeatures(degree=3, include_bias=False)
    poly_features = poly.fit_transform(X.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)
    y_predicted = poly_reg_model.predict(poly_features)

    plt.scatter(X, y, label="data")
    plt.plot(X, y_predicted, label="prediction", c="red")

    plt.legend()

    # Uncomment the following to create a png file of the plot
    # plt.savefig('../results/ml/pred4.png')

    plt.show()
