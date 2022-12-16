import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression

#################### Statistical and Visuals Functions ##################################
alpha = 0.05 
labels = ['PearsonsR', 'P-Value', 'Outcome']
def make_stats_df():
    '''
    Function creates dataframe for results of pearsonsr statistical 
    test for all features.
    '''
    results_stats_df = pd.DataFrame()
    results_stats_df['Index Scores'] = labels
    return results_stats_df

def get_stats_trust(train,results_stats_df):
    '''
    Function gets results of pearsonsr statistical test for 
    perceptions of corruptions in gov and happiness score.
    '''

    r, p = stats.pearsonr(train.Perceptions_Corruption_Gov, train.Happiness_Score)
    if p < alpha:
        print = (f'We reject the null hypothesis')
    else:
        print = (f'We fail to reject the null hypothesis')

    #print(f'pearsonsr test = {r:.4f}')
    results_stats_df['Perceptions of Corruption'] = r,p,print
    return results_stats_df

def get_stats_year(train, results_stats_df):
    '''
    Function gets results of pearsonsr statistical test for year
    and happiness score.
    '''

    r, p = stats.pearsonr(train.Year, train.Happiness_Score)
    if p < alpha:
        print = (f'We reject the null hypothesis')
    else:
        print = (f'We fail to reject the null hypothesis')

    #print(f'pearsonsr test = {r:.4f}')
    results_stats_df['Year'] = r,p,print
    return results_stats_df

def get_stats_health(train, results_stats_df):
    '''
    Function gets results of pearsonsr statistical test for health life
    expectancy and happiness score.
    '''

    r, p = stats.pearsonr(train.Health_Life_Expectancy, train.Happiness_Score)
    if p < alpha:
        print = (f'We reject the null hypothesis')
    else:
        print = (f'We fail to reject the null hypothesis')

    #print(f'pearsonsr test = {r:.4f}')
    results_stats_df['Health'] = r,p,print
    return results_stats_df

def get_stats_gen(train, results_stats_df):
    '''
    Function gets results of pearsonsr statistical test for generosity
    and happiness score.
    '''

    r, p = stats.pearsonr(train.Generosity, train.Happiness_Score)
    if p < alpha:
        print = (f'We reject the null hypothesis')
    else:
        print = (f'We fail to reject the null hypothesis')

    #print(f'pearsonsr test = {r:.4f}')
    results_stats_df['Generosity'] = r,p,print
    return results_stats_df

def get_stats_eco(train, results_stats_df):
    '''
    Function gets results of pearsonsr statistical test for economy
    and happiness score.
    '''

    r, p = stats.pearsonr(train.Economy, train.Happiness_Score)
    if p < alpha:
        print = (f'We reject the null hypothesis')
    else:
        print = (f'We fail to reject the null hypothesis')

    #print(f'pearsonsr test = {r:.4f}')
    results_stats_df['Economy'] = r,p,print
    return results_stats_df

def get_stats_free(train, results_stats_df):
    '''
    Function gets results of pearsonsr statistical test for freedom
    and happiness score.
    '''

    r, p = stats.pearsonr(train.Freedom, train.Happiness_Score)
    if p < alpha:
        print = (f'We reject the null hypothesis')
    else:
        print = (f'We fail to reject the null hypothesis')

    #print(f'pearsonsr test = {r:.4f}')
    results_stats_df['Freedom'] = r,p,print
    return results_stats_df

def get_results(train, results_stats_df):
    '''
    Function gets results of all pearsonsr statistical test
    and appends them to dataframe.
    '''
    results_stats_df = get_stats_health(train, results_stats_df)
    results_stats_df = get_stats_year(train, results_stats_df)
    results_stats_df = get_stats_gen(train, results_stats_df)
    results_stats_df = get_stats_eco(train, results_stats_df)
    results_stats_df = get_stats_free(train, results_stats_df)
    results_stats_df = get_stats_trust(train,results_stats_df) 
    return results_stats_df


    

