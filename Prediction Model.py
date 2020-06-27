# By: Kevin Wang
# Created: June 18th, 2020
### This is the Model Building Process
###ML models attempted were Decision Tree Classifier, Gradient Boosting Classifier, KNN and RandomForest


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Import Cleaned Data
df = pd.read_csv('google_playstore_cleaned.csv')

### Feature Engineering ###

#Filtering for relevant variables - Remove App Name
print(df.columns)
df = df.drop(['App Name'], axis = 1)
print(df.columns)

#Reduce number of target classes
df.loc[df['Installs'].isin(['0 - 100', '100 - 500', '500 - 1,000']), 'Installs'] = '0 - 1,000'
df.loc[df['Installs'].isin(['1,000 - 5,000', '5,000 - 10,000']), 'Installs'] = '1,000 - 10,000'
df.loc[df['Installs'].isin(['10,000 - 50,000', '50,000 - 100,000']), 'Installs']  = '10,000 - 100,000'
df.loc[df['Installs'].isin(['100,000 - 500,000', '500,000 - 1,000,000']), 'Installs']  = '100,000 - 1,000,000'
df.loc[df['Installs'].isin(['1,000,000 - 5,000,000', '5,000,000 - 10,000,000']), 'Installs']  = '1,000,000 - 10,000,000'
df.loc[df['Installs'].isin(['10,000,000 - 50,000,000', '50,000,000 - 100,000,000']), 'Installs']  = '10,000,000 - 100,000,000'
df.loc[df['Installs'].isin(['100,000,000 - 500,000,000', '500,000,000 - 1,000,000,000']), 'Installs']  = '100,000,000 - 1,000,000,000'
df.loc[df['Installs'].isin(['1,000,000,000 - 5,000,000,000', '5,000,000,000+']), 'Installs']  = '1,000,000,000+'


df.Installs = pd.Categorical(df.Installs, ['0 - 1,000','1,000 - 10,000', '10,000 - 100,000', '100,000 - 1,000,000', '1,000,000 - 10,000,000', '10,000,000 - 100,000,000', '100,000,000 - 1,000,000,000', '1,000,000,000+'])
print(df.Installs.value_counts().sort_index())
print(df.shape)


#One hot encoding due to Sklearn categorical variable limitation (Sklearn Decision trees treat categorical variable as continuous)
strat = df.Category.values
df = pd.get_dummies(df, columns=['Content Rating', 'Category', 'Game_genre'], drop_first=True)
print(df.columns)

#Train Test Split - Simple Random Sampling (with Stratification)
X = df.drop(['Installs'], axis = 1).values
y = df.Installs.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = strat, random_state = 42)
X.shape[0] == y.shape[0]


### Model Building ###

#Tuning with GridsearchCV - Best parameters are in comments and also in the Models below
#Random Forest

from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier()
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [20, 30, 40, 50], "max_features": ['sqrt']}


grid_search = GridSearchCV(rf, param_grid=param_grid)
grid_search.fit(X, y)

print("Best score is {}".format(grid_search.best_params_))
print("Best score is {}".format(grid_search.best_score_))
# Best parameters are {'max_depth': 20, 'max_features': 'sqrt', 'n_estimators': 200}

#Gradient Boosting Classifier
param_grid = {
    "learning_rate": [0.01,0.025, 0.05],
    "max_depth":[5,8,10],
    "n_estimators":[25, 50, 100],
    "verbose":[1]
    }

grid_search = GridSearchCV(gbm, param_grid=param_grid)
grid_search.fit(X, y)


print("Best score is {}".format(grid_search.best_params_))
print("Best score is {}".format(grid_search.best_score_))
# Best parameters are {'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 50, 'verbose': 1}





#Decision Tree Classifier Train
dt = DecisionTreeClassifier(max_depth = 8, max_features = 'sqrt')
dt.fit(X_train, y_train)
y_pred = dt.predict(X_train)

acc = accuracy_score(y_train, y_pred)
print("Decision Tree train data accuracy: {:.2f}".format(acc))

#Decision Tree Classifier 
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Decision Tree Test data accuracy: {:.2f}".format(acc))


# Decision Tree Cross Val
a = np.mean(cross_val_score(dt, X, y, scoring = 'accuracy', cv = 10))
print("Decision Tree cross validation accuracy: {:.2f}".format(a))

#The Decision Tree gave an accuracy of around 0.63

#RandomForest Classifier Train
rf = RandomForestClassifier(max_depth = 20, n_estimators = 100)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_train)

acc = accuracy_score(y_train, y_pred)
print("Random Forest Train data accuracy: {:.2f}".format(acc))


#RandomForest Classifier Test 
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Random Forest Test data accuracy: {:.5f}".format(acc))

#RandomForest Cross Val 
b= np.mean(cross_val_score(rf, X, y, scoring = 'accuracy', cv = 10))
print("Random Forest Cross validation accuracy : {:.2f}".format(b))

#Random Forest gave an Accuracy for test set and cross validation 0.73

#GBM Classifier
gbm = GradientBoostingClassifier(learning_rate = 0.05, max_depth = 8, n_estimators = 400, verbose =1)
gbm.fit(X_train,y_train)

y_pred = gbm.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Gradient Boosting Classifier Test data accuracy: {:.5f}".format(acc))

#GBM Gave an accuracy of 0.73

#K Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print('knn train data accuracy',knn.score(X_train, y_train))  
print('knn test data accuracy', knn.score(X_test,y_test))
print('knn cross validation accuracy', np.mean(cross_val_score(knn,X, y, cv = 5)))

#KNN gave an accuracy of 0.66

#Voting Classifier
classifiers = [('Random Forest', rf), ('Gradient Boosting Classifier', gbm)]
vc = VotingClassifier(estimators = classifiers)
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Voting Classifier Ensemble: {:.5f}".format(acc))

#Voting classifier with the two best gave an accuracy of 0.71



#Random Forest was the best model due to high accuracy and lowest computational time
