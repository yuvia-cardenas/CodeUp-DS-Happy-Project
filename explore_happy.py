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
def get_stats_trust(train):
    '''
    Function gets results of pearsonsr statistical test for 
    perceptions of corruptions in gov and happiness score.
    '''

    r, p = stats.pearsonr(train.Perceptions_Corruption_Gov, train.Happiness_Score)
    if p < alpha:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

    print(f'pearsonsr test = {r:.4f}')
     
def get_chi_year(train):
    '''
    Function gets results of chi-square statistical test for year
    and happiness score.
    '''

    observed = pd.crosstab(train.Happiness_Score, train.Year)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def get_stats_year(train):
    '''
    Function gets results of pearsonsr statistical test for year
    and happiness score.
    '''

    r, p = stats.pearsonr(train.Year, train.Happiness_Score)
    if p < alpha:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

    print(f'pearsonsr test = {r:.4f}')

def get_stats_health(train):
    '''
    Function gets results of pearsonsr statistical test for health life
    expectancy and happiness score.
    '''

    r, p = stats.pearsonr(train.Health_Life_Expectancy, train.Happiness_Score)
    if p < alpha:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

    print(f'pearsonsr test = {r:.4f}')

def get_stats_gen(train):
    '''
    Function gets results of pearsonsr statistical test for generosity
    and happiness score.
    '''

    r, p = stats.pearsonr(train.Generosity, train.Happiness_Score)
    if p < alpha:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

    print(f'pearsonsr test = {r:.4f}')

def get_stats_eco(train):
    '''
    Function gets results of pearsonsr statistical test for economy
    and happiness score.
    '''

    r, p = stats.pearsonr(train.Economy, train.Happiness_Score)
    if p < alpha:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

    print(f'pearsonsr test = {r:.4f}')

def get_stats_free(train):
    '''
    Function gets results of pearsonsr statistical test for freedom
    and happiness score.
    '''

    r, p = stats.pearsonr(train.Freedom, train.Happiness_Score)
    if p < alpha:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

    print(f'pearsonsr test = {r:.4f}')


def select_kbest(x_train, y_train):
    '''
    Function gets results of select kbest test for our train data
    '''
    kbest = SelectKBest(f_regression, k=3)
    _ = kbest.fit(x_train, y_train)
    kbest_results = pd.DataFrame(
        dict(p=kbest.pvalues_, f=kbest.scores_),
                                 index = x_train.columns)
    return kbest_results

    

