# Shelter-animal-outcome-prediction
Kaggle Practice dataset, Main focus on converting categorical variable into numeric variables.
Multi class classification problem.

# Basic data exploration:
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline
train = pd.read_csv('train.csv')
train['OutcomeType'].value_counts().plot.bar()

train.dtypes

columns = train.columns
for column in columns:
    print(column)
    print(train[column].nunique())
    
# Missing value handling:
train.apply(lambda x: sum(x.isnull()/len(train)))

train = train.drop('OutcomeSubtype', axis=1)

train['Name'] = train[['Name']].fillna(value=0)
train['has_name'] = (train['Name'] != 0).astype('int64')
train = train.drop('Name', axis=1)

train = train.apply(lambda x:x.fillna(x.value_counts().index[0]))
train.apply(lambda x: sum(x.isnull()/len(train)))

train = train.drop('AnimalID', axis=1)

# Handling high cardinality:

# 1) Most popular values
color_counts = train['Color'].value_counts()
color_others = set(color_counts[color_counts < 300].index)
train['top_colors'] = train['Color'].replace(list(color_others), 'Others')
print(train['top_colors'].nunique())

# 2) New features
import re
train['breed_type'] = train.Breed.str.extract('({})'.format('|'.join(['Mix'])), 
                        flags=re.IGNORECASE, expand=False).str.lower().fillna('pure')
                        
train['multi_colors'] = train['Color'].apply(lambda x : 1 if '/' in x else 0)
                        
# 3) Numerical representation
def age_converter(row):
    age_string = row['AgeuponOutcome']
    [age,unit] = age_string.split(" ")
    unit = unit.lower()
    if("day" in unit):
        if age=='0': return 1
        return int(age)
    if("week" in unit):
        if(age)=='0': return 7
        return int(age)*7
    elif("month" in unit):
        if(age)=='0': return 30
        return int(age) * 4*7
    elif("year" in unit):
        if(age)=='0': return 365
        return int(age) * 4*12*7
train['age_numeric'] = train.apply(age_converter, axis=1)
train = train.drop('AgeuponOutcome', axis=1)

# One hot encoding after basic data pre-processing and missing values treatment
train = train.drop(['Breed','Color', 'DateTime'], axis=1)
numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = train.select_dtypes(include=['object']).drop(['OutcomeType'], axis=1).columns
dummy_columns = pd.get_dummies(train[categorical_features])
final_train = pd.concat([dummy_columns, train],axis=1)
final_train = final_train.drop(['AnimalType', 'breed_type', 'SexuponOutcome', 'top_colors'], axis=1)

# Traning the classifier
X = final_train.drop('OutcomeType', axis=1)
y = final_train['OutcomeType']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                            random_state=0)
rf_model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
y_prob = rf_model.predict_proba(X_test)
print(log_loss(y_test, y_prob))

# feature importance
import numpy as np
features=X.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10,20))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show
