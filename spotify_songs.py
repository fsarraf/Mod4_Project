import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


#Data ETL
def etl(data):
    
    #create Dataframe and drop extra column
    df = pd.read_csv(data)
    df = df.drop('track_id', axis = 1)
    df['year'] = pd.DatetimeIndex(df.release_date).year
    #return dataframe
    return df

# Due to the nature of the dataset, there are a lot of tracks that are not suitable for our
# purpose of discovering what makes are the key factors that make a song popular. Among the 
# tracks we are removing are meditation and white noise tracks, as well as classical music tracks.

# We will remove these tracks in stages the first is by genres, we will keep the genres that we 
# would like to predict. Secondly we will remove any additional outliers, for example example an 
# songs that are less than a minute long as it is very rare to have songs that short.

def genres_reduction(df):

    df = df[(df.Genres == 'pop') |(df.Genres == 'rock') | (df.Genres == 'metal') | (df.Genres == 'hip hop') | (df.Genres == 'rap') | 
              (df.Genres == 'country') | (df.Genres == 'r&b') | (df.Genres == 'alternative') | (df.Genres == 'indie') | (df.Genres == 'edm') |
              (df.Genres == 'jazz') |(df.Genres == 'folk') |(df.Genres == 'reggae') ]
    
    return df

def variable_reduction(df):
    
    df = df[(df['danceability'] >= 0.3) & (df['danceability'] <= 0.8) & (df['acousticness'] >= 0) &
               (df['acousticness'] <= 0.4) & (df['energy'] >= 0.3) & (df['energy'] <= 1) & (df['loudness'] >= -20) &
               (df['instrumentalness'] <= 0.8) & (df['tempo'] >= 50) & (df['tempo'] <= 200) & (df['liveness'] <= 0.4) &
               (df['duration_ms'] <= 400000) & (df['duration_ms'] >= 100000) & (df['speechiness'] <= 0.2)]
    
    return df

# combining the above three functions into one, in addition to outputting a pairplot with predictor variable on the y-axis 

def clean_n_plot(data):
    
    df = etl(data)
    df_pass1 = genres_reduction(df)
    df_cleaned = variable_reduction(df_pass1)
    df_cleaned.head()
    
    # create pair plot
    
    sns.set(style="ticks")

    sns.pairplot(df_cleaned, hue="Genres", x_vars = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness',
                       'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence'], y_vars = ['popularity'], plot_kws = dict(alpha = 0.01))
    
    return df_cleaned

# creating a function that will output a dataframe of the r-squared and MSE of both linear and polynomial regressions for all the subgenres
# Due the fact that we have a lot of sub_genres with fewer than 30 songs  we decided not to test/train this model as low number of datapoint 
# will not give us an accurate result.
def sub_genres_regression(df):
    
    subgenres_list = list(df.subgenres.unique())
    
    sub_lst= []
    for sub in subgenres_list:
        df_sub = df[df['subgenres'] == sub]
    
        if df_sub.shape[0] > 30:

            y = df_sub['popularity']
            X = df_sub.drop(columns=['popularity', 'year', 'key', 'time_signature', 'artist_name', 'release_date', 'track_name', 'Genres', 'subgenres'], axis=1)

            reg = LinearRegression().fit(X, y)

            lin_mse = mean_squared_error(y, reg.predict(X))
            lin_r2 = r2_score(y, reg.predict(X))

            poly = PolynomialFeatures(3)
            X_fin = poly.fit_transform(X)

            reg_poly = LinearRegression().fit(X_fin, y)
            y_poly_pred = reg_poly.predict(X_fin)

            poly_mse = mean_squared_error(y, reg_poly.predict(X_fin))
            poly_r2 = r2_score(y, reg_poly.predict(X_fin))

            sub_lst.append({'sub': sub, 'lin_mse': lin_mse, 'lin_r2': lin_r2, 'poly_mse':poly_mse, 'poly_r2':poly_r2, 'no_rows':df_sub.shape[0]} )

    sub_df = pd.DataFrame(sub_lst)
    sub_df.sort_values(by=['poly_r2'], ascending=False, inplace=True)
        
    return sub_df
            
def genres_regression(df):
    genres_list = list(df.Genres.unique())

    genres_r2_3= []
    for genres in genres_list:
        df_gen = df[df['Genres'] == genres]

        if df_gen.shape[0] > 100:


            y = df_gen['popularity']
            X = df_gen.drop(columns=['popularity', 'year', 'key', 'time_signature', 'artist_name', 'release_date', 'track_name', 'Genres', 'subgenres'], axis=1)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            reg = LinearRegression().fit(X_train, y_train)

            y_hat_train = reg.predict(X_train)
            y_hat_test = reg.predict(X_test)

            train_mse = mean_squared_error(y_train, y_hat_train)
            test_mse = mean_squared_error(y_test, y_hat_test)
            lin_r2 = r2_score(y_train, reg.predict(X_train))
            lin_test_r2 = r2_score(y_test, reg.predict(X_test))

            poly = PolynomialFeatures(3)
            X_fin = poly.fit_transform(X_train)
            X_tes = poly.fit_transform(X_test)
            reg_poly = LinearRegression().fit(X_fin, y_train)
            y_train_poly_pred = reg_poly.predict(X_fin)
            y_test_poly_pred = reg_poly.predict(X_tes)

            poly_mse = mean_squared_error(y_train, y_train_poly_pred)
            test_mse = mean_squared_error(y_test, y_test_poly_pred)
            poly_r2 = r2_score(y_train, y_train_poly_pred)
            test_poly_r2 = r2_score(y_test, y_test_poly_pred)
            genres_r2_3.append({'genres': genres, 'lin_train_mse': train_mse, 'lin_test_mse': test_mse,  'lin_r2': lin_r2, 'lin_test_r2': lin_test_r2, 'poly_mse':poly_mse, 'test_mse':test_mse,'poly_r2':poly_r2, 
                                'test_poly_r2':test_poly_r2})

    genres_reg_df = pd.DataFrame(genres_r2_3)
    genres_reg_df.sort_values(by=['test_poly_r2'], ascending=False, inplace=True)
    return genres_reg_df

def genres_by_year_reg(df):
    
    music_by_year = df.groupby(['year'])
    genres_list = list(df.Genres.unique())
    
    genres_year= []
    for group, df_group in music_by_year:
        
        for genres in genres_list:
            df_gen = df_group[df_group['Genres'] == genres]

            if df_gen.shape[0] > 100:


                y = df_gen['popularity']
                X = df_gen.drop(columns=['popularity', 'year', 'key', 'time_signature', 'artist_name', 'release_date', 'track_name', 'Genres', 'subgenres'], axis=1)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                reg = LinearRegression().fit(X_train, y_train)

                y_hat_train = reg.predict(X_train)
                y_hat_test = reg.predict(X_test)

                train_mse = mean_squared_error(y_train, y_hat_train)
                test_mse = mean_squared_error(y_test, y_hat_test)
                lin_r2 = r2_score(y_train, reg.predict(X_train))
                lin_test_r2 = r2_score(y_test, reg.predict(X_test))
                coef = reg.coef_ 

                poly = PolynomialFeatures(3)
                X_fin = poly.fit_transform(X_train)
                X_tes = poly.fit_transform(X_test)
                reg_poly = LinearRegression().fit(X_fin, y_train)
                y_train_poly_pred = reg_poly.predict(X_fin)
                y_test_poly_pred = reg_poly.predict(X_tes)

                poly_mse = mean_squared_error(y_train, y_train_poly_pred)
                test_mse = mean_squared_error(y_test, y_test_poly_pred)
                poly_r2 = r2_score(y_train, y_train_poly_pred)
                test_poly_r2 = r2_score(y_test, y_test_poly_pred)
                
                genres_year.append({'year':group,'genres': genres, 'lin_train_mse': train_mse, 'lin_test_mse': test_mse,  'lin_r2': lin_r2, 'lin_test_r2': lin_test_r2,  'poly_mse':poly_mse, 'test_mse':test_mse,'poly_r2':poly_r2, 'test_poly_r2':test_poly_r2})

        
    genres_year_df = pd.DataFrame(genres_year)
    genres_year_df.sort_values(by=['test_poly_r2'], ascending=False, inplace=True)
    return genres_year_df

def get_sample(data, n):
    sample = []
    while len(sample) != n:
        x = np.random.choice(data)
        sample.append(x)
    
    return sample

def get_sample_mean(sample):
    return sum(sample) / len(sample)


def create_sample_distribution(data, dist_size=1000, n=1000):
    sample_dist = []
    while len(sample_dist) != dist_size:
        sample = get_sample(data, n)
        sample_mean = get_sample_mean(sample)
        sample_dist.append(sample_mean)
    
    return sample_dist

def variable_distributions(df):
    x_vars = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness',
                       'loudness', 'speechiness', 'tempo', 'valence']
    lst_dfs = []
    means_lst = []
    for year in range(1999,2020):
        print(year)
        dist_df = pd.DataFrame()

        for x in x_vars:
            print(x)
            df = music[music['year']== year]

            dist =  create_sample_distribution(df[x])
            dist_df[x]= dist
            dist_df['year']= year
            lst_dfs.append(dist_df)

            means_lst.append({'year': year, x : np.mean(dist)})

    dfs_concat= pd.concat(lst_dfs)
    
    return dfs_concat

def rap_coef(df):
   
    df_gen = df[df['Genres'] == 'rap']

    if df_gen.shape[0] > 100:


        y = df_gen['popularity']
        X = df_gen.drop(columns=['popularity', 'year', 'key', 'time_signature', 'artist_name', 'release_date', 'track_name', 'Genres', 'subgenres'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = LinearRegression().fit(X_train, y_train)

        y_hat_train = reg.predict(X_train)
        y_hat_test = reg.predict(X_test)

        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)
        lin_r2 = r2_score(y_train, reg.predict(X_train))
        lin_test_r2 = r2_score(y_test, reg.predict(X_test))

        coef = reg.coef_
        print(X_train.columns)
    
    
    return coef
