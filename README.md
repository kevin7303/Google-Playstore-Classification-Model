# Data-Science-Google-Play-Store-ML-Project
Project focused on: Data cleaning, Exploratory Data Analysis, Feature Engineering and Machine Learning Model Building

# Google Play Store Installs Predictor - Project Overview 
* Created a Machine Learning Model to estimate number of downloads an app on the Google Play Store will have.
* Cleaned and analyzed data provided by Gautham Prakash and Jithin Koshy at Kaggle.com with over 200,000 apps and 11 features.
* Provided detail visual analysis of the Google Play store to reach relational insight between features
* Engineered features such as Game Genre and used Category based imputation techniques. 
* Optimized Random Forest, Gradient Boosting Classifier, Decision Tree and KNN using GridsearchCV to reach the best model. 
 

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn  
**Original Kaggle Dataset:** https://www.kaggle.com/gauthamp10/google-playstore-apps?select=Google-Playstore-Full.csv

## Data Overview
Data contains 267,052 rows and 11 columns:
Columns (features) are:
*	App Name
*	Category
*	Rating
*	Reviews
*	Installs 
*	Size
*	Price 
*	Content Rating
*	Last Updated
*	Minimum Version
* Latest Version

## Data Cleaning
I did extensive data cleaning in order to facilitate the exploratory analysis and the model building process:

*	Fixed data scraping error and removed additional blank columns created
* Corrected misalignment by shifting the affected feature values to their correct column
*	Coerced the features to their respective correct data types after handling string replacement and unique complications
*	Performed data imputation based on the Category feature in order to replace null values with the closest value to the true value
*	Dropped irrelevant features such as Minimum Version and Latest Version
*	Changed target variable classes and ordered them
*	Aggregated all game genre as the Game category and created a new game genre feature

## EDA
The expoloraty data analysis was done to better visualize and understand data set before undergoing the model building process.

Explored the data and the categorical distributions. Created plots to visually display feature relations and possible correlations:
Below are some of the graphs created with seaborn:

![alt text](https://github.com/kevin7303/Data-Science-Google-Play-Store-ML-Project/blob/master/Apps%20per%20bracket.png "Number of Apps per Install Bracket")
![alt text](https://github.com/kevin7303/Data-Science-Google-Play-Store-ML-Project/blob/master/Apps%20per%20category.png "Number of Apps per Category")
![alt text](https://github.com/kevin7303/Data-Science-Google-Play-Store-ML-Project/blob/master/App%20Ratings%20based%20on%20Price%20and%20Category.png "App Ratings by Price and Category")

## Model Building 
I wanted to create a model that would make meaningful and accurate predictions for aspiring app creators to know what features are the most important when maximizing installs.


**Restructured the test data labels to better distribute the classes and create meaning full differences between classification class.**
*The target label breakdowns and count were:*
* Installs Brackets
* 0 - 1,000                      
* 1,000 - 10,000                 
* 10,000 - 100,000               
* 100,000 - 1,000,000            
* 1,000,000 - 10,000,000         
* 10,000,000 - 100,000,000        
* 100,000,000 - 1,000,000,000      
* 1,000,000,000+                    


Performed One hot encoding on the categorical variables in order to accomodate Sklearn Decision trees treatment of categorical variables as continuous

Split the data set into  train test split of 70/30, stratified based on Category.

The data would be judged based on the measure of Accuracy.

I tried 4 different models
*	**Decision Tree Classifier** – Baseline for the model
*	**K Nearest Neighbors** – A second baseline reference to understand what accuracy the model was reaching
*	**Random Forest** – Random Forest was chosen due to its ability handle multiclassification problems and its reliability  
*	**Gradient Boosting Classifier** – GBC was used to measure the difference in accuracy between Random Forest and a more complex, and intensive computing model

I also used the Voting Classifier ensemble method in an attempt to aggregate the strengths of Random Forest and Gradient Boosting Classifier.

## Model performance
Tuning was done on Random Forest and Gradient Boosting Classifier as their Default parameters compared to the two baseline models showed significant improvement 

**Base Models:
*	**Decision Tree Classifier** – Accuracy: 0.63
*	**K Nearest Neighbors** – Accuracy: 0.66

**Tuned Models
*	**Random Forest** – Accuracy: 0.73 

*Best parameters are {'max_depth': 20, 'max_features': 'sqrt', 'n_estimators': 200}*

*	**Gradient Boosting Classifier** – Accuracy: 0.73

*Best parameters are {'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 50}*

**Ensemble method
* **Voting Classifier** - Accuracy: 0.71

**A tuned Random Forest was the best model due to high accuracy and lowest computational time**


