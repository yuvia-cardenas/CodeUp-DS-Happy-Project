'''Aquire and Prep Happy data'''
import pandas as pd
import numpy as np
import os


###################################################
################## ACQUIRE DATA ###################
###################################################

'''Reads happy into a pandas DataFrame from CSV'''
def get_happy_2015():
    filename = "2015.csv"

    os.path.isfile(filename)
    return pd.read_csv(filename)


def get_happy_2016():
    filename = "2016.csv"

    os.path.isfile(filename)
    return pd.read_csv(filename)

def get_happy_2017():
    filename = "2017.csv"

    os.path.isfile(filename)
    return pd.read_csv(filename)

def get_happy_2018():
    filename = "2018.csv"

    os.path.isfile(filename)
    return pd.read_csv(filename)

def get_happy_2019():
    filename = "2019.csv"

    os.path.isfile(filename)
    return pd.read_csv(filename)
    
###################################################
################## PREPARE DATA ###################
###################################################

def wrangle_happi():
    '''
    Reads happy into a pandas DataFrame from CSV
    drop columns, drop any rows with Null values,
    rename columns, and return cleaned DataFrames.
    '''
    # Acquire data 
    df_2015 = get_happy_2015()
    df_2016 = get_happy_2016()
    df_2017 = get_happy_2017()
    df_2018 = get_happy_2018()
    df_2019 = get_happy_2019()

    # Drop columns
    df_2015.drop(columns=['Region', 'Standard Error', 'Dystopia Residual'], inplace=True)
    df_2016.drop(columns=['Region', 'Lower Confidence Interval', 'Upper Confidence Interval','Dystopia Residual'], inplace=True)
    df_2017.drop(columns=['Whisker.high', 'Whisker.low','Dystopia.Residual'], inplace=True)
    
    # Rename columns

    df_2015 = df_2015.rename(columns={'Family':'Social Support'})
    df_2015 = df_2015.rename(columns={'Happiness Score':'Happiness_Score'})
    df_2015 = df_2015.rename(columns={'Happiness Rank':'Happiness_Rank'})
    df_2015 = df_2015.rename(columns={'Trust (Government Corruption)':'Perceptions_Corruption_Gov'})
    df_2015 = df_2015.rename(columns={'Health (Life Expectancy)':'Health_Life_Expectancy'})
    df_2015 = df_2015.rename(columns={'Economy (GDP per Capita)':'Economy'})

    df_2016 = df_2016.rename(columns={'Economy (GDP per Capita)':'Economy'})
    df_2016 = df_2016.rename(columns={'Health (Life Expectancy)':'Health_Life_Expectancy'})
    df_2016 = df_2016.rename(columns={'Family':'Social Support'})
    df_2016 = df_2016.rename(columns={'Happiness Score':'Happiness_Score'})
    df_2016 = df_2016.rename(columns={'Happiness Rank':'Happiness_Rank'})
    df_2016 = df_2016.rename(columns={'Trust (Government Corruption)':'Perceptions_Corruption_Gov'})
  

    df_2017 = df_2017.rename(columns={'Happiness.Score':'Happiness_Score'})
    df_2017 = df_2017.rename(columns={'Happiness.Rank':'Happiness_Rank'})
    df_2017 = df_2017.rename(columns={'Family':'Social Support'})
    df_2017 = df_2017.rename(columns={'Economy..GDP.per.Capita.':'Economy'})
    df_2017 = df_2017.rename(columns={'Health..Life.Expectancy.':'Health_Life_Expectancy'})
    df_2017 = df_2017.rename(columns={'Trust..Government.Corruption.':'Perceptions_Corruption_Gov'})
  

    df_2018 = df_2018.rename(columns={'Country or region':'Country'})
    df_2018 = df_2018.rename(columns={'Healthy life expectancy':'Health_Life_Expectancy'})
    df_2018 = df_2018.rename(columns={'Overall rank':'Happiness_Rank'})
    df_2018 = df_2018.rename(columns={'Score':'Happiness_Score'})
    df_2018 = df_2018.rename(columns={'GDP per capita':'Economy'})
    df_2018 = df_2018.rename(columns={'Perceptions of corruption':'Perceptions_Corruption_Gov'})
    df_2018 = df_2018.rename(columns={'Freedom to make life choices':'Freedom'})

    df_2019 = df_2019.rename(columns={'Freedom to make life choices':'Freedom'})
    df_2019 = df_2019.rename(columns={'Country or region':'Country'})
    df_2019 = df_2019.rename(columns={'Healthy life expectancy':'Health_Life_Expectancy'})
    df_2019 = df_2019.rename(columns={'Overall rank':'Happiness_Rank'})
    df_2019 = df_2019.rename(columns={'Score':'Happiness_Score'})
    df_2019 = df_2019.rename(columns={'GDP per capita':'Economy'})
    df_2019 = df_2019.rename(columns={'Perceptions of corruption':'Perceptions_Corruption_Gov'})
    
    # Reindex columns to match
    column_titles = ['Country', 'Happiness_Rank', 'Happiness_Score', 'Economy', 'Social Support', 'Health_Life_Expectancy', 'Freedom', 'Perceptions_Corruption_Gov', 'Generosity']
    df_2017 = df_2017.reindex(columns=column_titles)
    df_2018 = df_2018.reindex(columns=column_titles)
    df_2019 = df_2019.reindex(columns=column_titles)

    # Adding Year columns
    df_2015['Year']= 2015
    df_2016['Year']= 2016
    df_2017['Year']= 2017
    df_2018['Year']= 2018
    df_2019['Year']= 2019

    return df_2015, df_2016, df_2017, df_2018, df_2019

    
def join_happy(df_2015, df_2016, df_2017, df_2018, df_2019):
    happy_data = [df_2015, df_2016, df_2017, df_2018, df_2019]
    happy_df = pd.concat(happy_data)
    happy_df.drop(columns=['Social Support'], inplace=True)
    happy_df.fillna(happy_df.mean(numeric_only=True).round(1), inplace=True)

    return happy_df


