import pandas as pd
import numpy as np
import os

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TweedieRegressor

###################################################
################## SPLITS DATA ####################
###################################################

def split_data(happy_df):
    '''
    takes in a DataFrame and returns split train, validate, and test DataFrames.
    '''
    
    train_validate, test = train_test_split(happy_df, test_size=.2, random_state=123)
    
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    return train, validate, test

###################################################
################## Modeling Functions #############
###################################################

def model_sets(train,validate,test):
    '''
    Function drops the target of Happiness_Score and associated Happiness_Rank columns 
    then splits data into 
    predicting variables (x) and target variable (y)
    ''' 

    x_train = train.drop(columns=['Happiness_Score', 'Happiness_Rank', 'Country'])
    y_train = train.Happiness_Score


    x_validate = validate.drop(columns=['Happiness_Score', 'Happiness_Rank','Country'])
    y_validate = validate.Happiness_Score

    x_test = test.drop(columns=['Happiness_Score', 'Happiness_Rank', 'Country'])
    y_test = test.Happiness_Score

    return x_train, y_train, x_validate, y_validate, x_test, y_test

def predict(train, validate, test):
    '''
    Function takes the baseline and creates dataframe
    '''
    train_predictions = pd.DataFrame({
    'Happiness_Score': train.Happiness_Score})
    train_predictions["Baseline"]= train.Happiness_Score.mean()
    validate_predictions = pd.DataFrame({
    'Happiness_Score': validate.Happiness_Score})
    validate_predictions["Baseline"]= train.Happiness_Score.mean()
    test_predictions = pd.DataFrame({
    'Happiness_Score': test.Happiness_Score})
    
    return train_predictions, validate_predictions, test_predictions

def simple_lm_model(train, x_train, y_train, validate, x_validate, train_predictions, validate_predictions):
    '''
    Function takes the predicted variable (x) and target variable (y)
    fits them to the simple linear regression model and computes the predictions 
    and then adds them to the dataframe  
    ''' 
    lm_model = LinearRegression().fit(x_train, y_train)
    train_predictions['lm_predictions'] = lm_model.predict(x_train)
    train['lm_predictions'] = lm_model.predict(x_train)
    validate_predictions['lm_predictions'] = lm_model.predict(x_validate)
    validate['lm_predictions'] = lm_model.predict(x_validate)

    lm_co = lm_model.coef_
    lm_int = lm_model.intercept_

    simp_co = pd.Series(lm_model.coef_, index=x_train.columns).sort_values()
    print(simp_co)
    return train, train_predictions, validate, validate_predictions

def glm_model(train, x_train, y_train, validate, x_validate, train_predictions, validate_predictions):
    '''
    Function takes the predicted variable (x) and target variable (y)
    fits them to the generalized linear regression model and computes the predictions 
    and then adds them to the dataframe  
    '''

    glm_model = TweedieRegressor(power=0, alpha=1).fit(x_train, y_train)
    train_predictions['glm_predictions'] = glm_model.predict(x_train)
    train['glm_predictions'] = glm_model.predict(x_train)
    validate_predictions['glm_predictions'] = glm_model.predict(x_validate)
    validate['glm_predictions'] = glm_model.predict(x_validate)

    glm_co = pd.Series(glm_model.coef_, index=x_train.columns).sort_values()
    print(glm_co)
    return train, train_predictions, validate, validate_predictions

###################################################
################## Evaluate Models on Train #######
###################################################

def baseline_mean_errors(train):
    
    train['Baseline']= train.Happiness_Score.mean()
    y = train.Happiness_Score
    yhat = train.Baseline
    
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
      "\nRMSE: ", round(RMSE, 2))

    
def lm_errors(train):
    y = train.Happiness_Score
    yhat = train.lm_predictions

    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
      "\nRMSE: ", round(RMSE, 2))


def glm_errors(train):
    y = train.Happiness_Score
    yhat = train.glm_predictions
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
      "\nRMSE: ", round(RMSE, 2))

def test_lm_model(test, x_test, y_test, validate, x_validate, test_predictions, validate_predictions):
    '''
    Function takes the predicted variable (x) and target variable (y)
    fits them to the simple linear regression model and computes the predictions 
    and then adds them to the dataframe  
    ''' 
    lm_model = LinearRegression().fit(x_test, y_test)
    test_predictions['lm_predictions'] = lm_model.predict(x_test)
    test['lm_predictions'] = lm_model.predict(x_test)
    
    lm_co = lm_model.coef_
    lm_int = lm_model.intercept_

    simp_co = pd.Series(lm_model.coef_, index=x_test.columns).sort_values()
    print(simp_co)
    return test, test_predictions, validate, validate_predictions

def lm_test_errors(test):
    y = test.Happiness_Score
    yhat = test.lm_predictions

    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    print("SSE:", round(SSE, 2),"\nMSE: ", round(MSE, 2), 
      "\nRMSE: ", round(RMSE, 2))

def make_metric_df(y, y_hat, model_name, metric_df):
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
                'model': model_name, 
                'RMSE': mean_squared_error(
                    y,
                    y_hat) ** .5,
                'r^2': explained_variance_score(
                    y,
                    y_hat)
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name, 
                'RMSE': mean_squared_error(
                    y,
                    y_hat) ** .5,
                'r^2': explained_variance_score(
                    y,
                    y_hat)
            }, ignore_index=True)