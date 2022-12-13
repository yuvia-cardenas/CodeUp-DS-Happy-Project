import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LassoLars
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

    x_train = train.drop(columns=['Happiness_Score', 'Happiness_Rank'])
    y_train = train.Happiness_Score


    x_validate = validate.drop(columns=['Happiness_Score', 'Happiness_Rank'])
    y_validate = validate.Happiness_Score

    x_test = test.drop(columns=['Happiness_Score', 'Happiness_Rank'])
    y_test = test.Happiness_Score

    return x_train, y_train, x_validate, y_validate, x_test, y_test

def predictions(train, validate):
    '''
    Function takes the baseline and creates dataframe
    '''
    train_predictions = pd.DataFrame({
    'Happiness_Score': train.Happiness_Score})
    validate_predictions = pd.DataFrame({
    'Happiness_Score': validate.Happiness_Score})
    return train_predictions, validate_predictions

def simple_lm_model(x_train, y_train, x_validate, train_predictions, validate_predictions):
    '''
    Function takes the predicted variable (x) and target variable (y)
    fits them to the simple linear regression model and computes the predictions 
    and then adds them to the dataframe  
    ''' 
    lm_model = LinearRegression().fit(x_train, y_train)
    train_predictions['lm_predictions'] = lm_model.predict(x_train)
    validate_predictions['lm_predictions'] = lm_model.predict(x_validate)

    lm_co = lm_model.coef_
    lm_int = lm_model.intercept_

    simp_co = pd.Series(lm_model.coef_, index=x_train.columns).sort_values()
    print(simp_co)
    return train_predictions, validate_predictions

def lasso_model(x_train, y_train, x_validate, train_predictions, validate_predictions):
    '''
    Function takes the predicted variable (x) and target variable (y)
    fits them to the lasso lars regression model and computes the predictions 
    and then adds them to the dataframe  
    ''' 

    lars = LassoLars(alpha=1).fit(x_train, y_train)
    train_predictions['lars_predictions'] = lars.predict(x_train)
    validate_predictions['lars_predictions'] = lars.predict(x_validate)

    lars_co = pd.Series(lars.coef_, index=x_train.columns).sort_values()
    print(lars_co)
    return train_predictions, validate_predictions

def glm_model(x_train, y_train, x_validate, train_predictions, validate_predictions):
    '''
    Function takes the predicted variable (x) and target variable (y)
    fits them to the generalized linear regression model and computes the predictions 
    and then adds them to the dataframe  
    '''

    glm_model = TweedieRegressor(power=0, alpha=1).fit(x_train, y_train)
    train_predictions['glm_predictions'] = glm_model.predict(x_train)
    validate_predictions['glm_predictions'] = glm_model.predict(x_validate)

    glm_co = pd.Series(glm_model.coef_, index=x_train.columns).sort_values()
    print(glm_co)
    return train_predictions, validate_predictions

###################################################
################## Evaluate Models on Train #######
###################################################

def baseline_mean_errors(train):
    y = train.Happiness_Score
    baseline = train.Happiness_Score.mean()
    
    MSE = mean_squared_error(y, baseline)
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
    

def lars_errors(train):
    y = train.Happiness_Score
    yhat = train.lars_predictions
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
