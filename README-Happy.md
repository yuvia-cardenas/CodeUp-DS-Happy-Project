# CodeUp-DS-Happy-Project
 
### Project Goals
* This project is based on the World Happiness Report.
* The World Happiness Report is a survey that asks people to rate their happiness on a scale based on questions using a scale from 1-10.
* My goal is to use the features from the data gathered to create a model that can effectly predict Happiness.
### The Plan
* Aquire data from the Kaggle database
* Prepare data for exploration by creating tailored columns from existing data
#### Explore data in search of key features with the basic following questions:
* What is average score for Happiness over the 5 year span?
* What key features have the best significance?
#### Develop a Model to predict happiness score
* Use key features identified to build predictive models of different types
* Evaluate models on train and validate data samples
* Select the best model based on RSME
* Evaluate the best model on test data samples
#### Draw conclusions

### Steps to Reproduce
* Clone this repo.
* Acquire the data from Kagle database
* Put the data in the file containing the cloned repo.
* Run notebook
### Conclusions
* SimpleLinear Regression model RMSE scores:

        * 0.562486 on training data samples
        * 0.593547 on validate data samples
        * 0.294145 on test data samples
#### Key TakeAway:
    SimpleLinear Regression model was successful on all train, validate and test data sets. 

### Recommendations
 * Consider age of persons contributing as a feature  
 * Consider gender of persons contributing as a feature
 * Consider gathering data seasonally