# By: Kevin Wang
# Created: June 16th, 2020
### This is the Data Clean Up Process for the Google Playstore Installation Prediction Model
### The dataset is graciously provided by Gautham Prakash and Jithin Koshy at Kaggle and is linked below
### https://www.kaggle.com/gauthamp10/google-playstore-apps?select=Google-Playstore-Full.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('Google-Playstore-Full.csv')

#Looking through the Raw Data: Null Values and Data Types
print(df.columns)
print(df.isna().sum())
print(df.count())
print(df.dtypes)
print(df.describe())


#Data scraping error, additional columns created by accident. Removing all rows that have values in those columns
rows_drop= df[df['Unnamed: 11'].notnull()].index.values.tolist()
print(df[df['Unnamed: 11'].notnull()].index.values)
# print(rows_drop)
df_complete = df.drop(rows_drop)

#Drop unnamed columns
df_complete.drop(columns=['Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14'], inplace = True)

#Drop explicit NaN Rows
df_complete.dropna(inplace = True)

#Check data for complete values
print(df_complete.count())
print(df_complete.isna().sum())
print(df_complete.dtypes)

#Change rating to data type Float
df_complete.Rating = df_complete.Rating.astype('float')

#Fix price formating and change to data type float
df_complete.Price = df_complete.Price.str.replace('$', '')
df_complete.Price = df_complete.Price.astype('float')

#Fixing formating of Size of application by removing all M's
print(df_complete.Size.head())


#Edge case of a value with 1015k 
df_complete.Size = df_complete.Size.str.replace(',', ''
df_complete.Size = df_complete.Size.apply(lambda x: float(x[:-1])*0.001 if 'k' in x else x)


df_complete.Size = df_complete.Size.str.replace('M', '')
df_complete.Size = df_complete.Size.replace('Varies with device', np.NaN)
df_complete.Size = df_complete.Size.astype('float')

#Looking at Size based on Category for Imputation
print(df_complete.groupby('Category')['Size'].mean())

#Putting Travel category under the broad category of Travel_and_local
travel_index = df_complete[df_complete['Category'] == 'TRAVEL'].index.values
print(travel_index)
df_complete.loc[travel_index, 'Category'] = 'TRAVEL_AND_LOCAL'

#Impute the mean of each Category Group to the NA values of the respective groups
print(df_complete[df_complete['Size'].isna()].index.values)
print(df_complete.Size)
print(df_complete.Size.isna().sum())
df_complete['Size'] = df_complete.groupby('Category')['Size'].transform(lambda x: x.fillna(x.mean()))

#Drop Last Updated, Minimum Version and Latest Version
df_complete = df_complete.drop(['Last Updated', 'Minimum Version', 'Latest Version'], axis = 1)

#Turn Reviews into data type Float
df_complete.Reviews = df_complete.Reviews.astype('float')

#Check the target variable
print(df_complete.Installs.value_counts())

#Change target variable to category and give order
df_complete.loc[df_complete['Installs'].isin(['0+', '1+', '5+', '10+', '50+']), 'Installs'] = '0 - 100'
df_complete.loc[df_complete['Installs'] == '100+', 'Installs'] = '100 - 500'
df_complete.loc[df_complete['Installs'] == '500+', 'Installs'] = '500 - 1,000'
df_complete.loc[df_complete['Installs'] == '1,000+', 'Installs'] = '1,000 - 5,000'
df_complete.loc[df_complete['Installs'] == '5,000+', 'Installs'] = '5,000 - 10,000'
df_complete.loc[df_complete['Installs'] == '10,000+', 'Installs'] = '10,000 - 50,000'
df_complete.loc[df_complete['Installs'] == '50,000+', 'Installs'] = '50,000 - 100,000'
df_complete.loc[df_complete['Installs'] == '100,000+', 'Installs'] = '100,000 - 500,000'
df_complete.loc[df_complete['Installs'] == '500,000+', 'Installs'] = '500,000 - 1,000,000'
df_complete.loc[df_complete['Installs'] == '1,000,000+', 'Installs'] = '1,000,000 - 5,000,000'
df_complete.loc[df_complete['Installs'] == '5,000,000+', 'Installs'] = '5,000,000 - 10,000,000'
df_complete.loc[df_complete['Installs'] == '10,000,000+', 'Installs'] = '10,000,000 - 50,000,000'
df_complete.loc[df_complete['Installs'] == '50,000,000+', 'Installs'] = '50,000,000 - 100,000,000'
df_complete.loc[df_complete['Installs'] == '100,000,000+', 'Installs'] = '100,000,000 - 500,000,000'
df_complete.loc[df_complete['Installs'] == '500,000,000+', 'Installs'] = '500,000,000 - 1,000,000,000'
df_complete.loc[df_complete['Installs'] == '1,000,000,000+', 'Installs'] = '1,000,000,000 - 5,000,000,000'
df_complete.loc[df_complete['Installs'] == '5,000,000,000+', 'Installs'] = '5,000,000,000+'

df_complete.Installs = pd.Categorical(df_complete.Installs, ['0 - 100','100 - 500', '500 - 1,000', '1,000 - 5,000', '5,000 - 10,000', '10,000 - 50,000', '50,000 - 100,000', '100,000 - 500,000', '500,000 - 1,000,000','1,000,000 - 5,000,000', '5,000,000 - 10,000,000', '10,000,000 - 50,000,000', '50,000,000 - 100,000,000','100,000,000 - 500,000,000', '500,000,000 - 1,000,000,000', '1,000,000,000 - 5,000,000,000', '5,000,000,000+'])


# df_complete.Installs = df_complete.Installs.str.replace(r'^1+$', '0-50', regex=True)
print(df_complete.Installs.value_counts().sort_index())


# df_complete.Installs = df_complete['Installs'].str.replace(',', '')

#Create a new Games column with game category
games = df_complete['Category'].str.contains('GAME')
df_complete['Game_genre'] = df_complete.loc[games, 'Category']
print(df_complete['Game_genre'])
df_complete['Game_genre'] = df_complete['Game_genre'].str.replace('GAME_', '')

#Merge all Game types into Game under Category
df_complete.loc[games, 'Category'] = 'GAME'

#Make sure New Column Game_genre matches number of rows in Game Category
print(df_complete.Game_genre.count())
print(df_complete.Category.value_counts())


#Save new CSV
df_complete.to_csv('google_playstore_cleaned.csv', index = False)
