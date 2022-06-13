#! usr/bin/python3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

## load dataset
df_penguin_data = pd.read_csv("./2022-a1-penguin-classification.csv")
df_penguin_data.head()

# convert data to a binary classification problem
df_penguin_data["is_chinstrap"] = np.where((df_penguin_data["species"] == 'Chinstrap'),1, 0)
df_penguin_data = df_penguin_data.drop(columns=["species"], errors="ignore")

## Exploratory data analysis
# distribution of data in the target class
ax = df_penguin_data["is_chinstrap"].hist(grid=False, color="teal") # plot histogram of target class
ax.set_ylabel("Number of Penguins")
ax.set_title("Distribution of Chinstrap species");

# correlations between the numerical attributes and target variable
# correlation matrix
corr_matrix = df_penguin_data.corr()
corr_matrix["is_chinstrap"].sort_values(ascending=False)

# visualizations
# function to make boxplot for the features
def viz(data,attr, target):
    ax = data.boxplot(attr,target) 
    ax.set_title(" ")
    ax.get_figure().suptitle("Boxplot of feature vs target variable")
    ax.set_ylabel(attr)

features = ["culmen_length_mm","culmen_depth_mm","flipper_length_mm","body_mass_g","sex"]
for item in features:
    if item == "sex":
        CrosstabResult=pd.crosstab(index=df_penguin_data['sex'],columns=df_penguin_data['is_chinstrap'])
        CrosstabResult.plot.bar(ylabel="number of penguins")
    else:
        viz(df_penguin_data,item,"is_chinstrap")
		
## Outliers
# code to identify outliers based on the boxplot 
outlier1 = df_penguin_data[(df_penguin_data['culmen_length_mm'] >58) & (df_penguin_data['is_chinstrap'] == 0)]
outlier2 = df_penguin_data[(df_penguin_data['body_mass_g'] >4700) & (df_penguin_data['is_chinstrap'] == 1)]
outlier3 = df_penguin_data[(df_penguin_data['body_mass_g'] < 2800) & (df_penguin_data['is_chinstrap'] == 1)]
pd.concat([outlier1,outlier2,outlier3])

# code to remove outliers
df_penguin_data= df_penguin_data.drop(labels=[253,189,190], axis=0)

## Train/Test split
# drop all rows with missing values in "sex" column as we cannot impute the gender.
df_penguin_data = df_penguin_data.dropna(subset=['sex'], how='any')
df_penguin_data.describe()

X = df_penguin_data[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g','sex']] 
y = df_penguin_data['is_chinstrap']  # labels
# generate a stratified data set
from sklearn.model_selection import StratifiedShuffleSplit 
splitter=StratifiedShuffleSplit(n_splits=1,random_state=42)
for train,test in splitter.split(X,y):     # this will split the index
    X_train = X.iloc[train]
    y_train = y.iloc[train]
    X_test = X.iloc[test]
    y_test = y.iloc[test]
print(y_train.value_counts()) 
print(y_test.value_counts())  

## Baseline and heuristic models
from sklearn.dummy import DummyClassifier
strategies = ['most_frequent', 'uniform', 'constant']
test_scores = []
for s in strategies:
    if s =='constant':
        dclf = DummyClassifier(strategy = s, random_state = 42, constant = 0)
    else:
        dclf = DummyClassifier(strategy = s, random_state = 42)
    dclf.fit(X_train, y_train)
    score = dclf.score(X_test, y_test)
    test_scores.append(score)
test_scores
# visualize test scores of the models
ax = sns.stripplot(strategies, test_scores);
ax.set(xlabel ='Strategy', ylabel ='Test Score', title='Accuracy of baseline, random and constant models')
plt.show()

# F1 scores of models
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
strategies = ['most_frequent', 'uniform', 'constant']
F1_scores = []
for s in strategies:
    if s =='constant':
        dclf = DummyClassifier(strategy = s, random_state = 42, constant = 0) # predicts 0 / non_Chinstrap every time
    else:
        dclf = DummyClassifier(strategy = s, random_state = 42)
    y_train_pred = cross_val_predict( dclf, X_train, y_train, cv=3)
    f1 = f1_score(y_train, y_train_pred)
    F1_scores.append(f1)
F1_scores
# plots of F1 scores
ax = sns.stripplot(strategies, F1_scores);
ax.set(xlabel ='Strategy', ylabel ='F1 Score', title='F1 scores for baseline, random and constant models')
plt.show()

## Preprocessing data for machine learning
### preprocess and transform training data
training_data = X_train.copy() # make a copy of the training data
training_labels = y_train.copy() # make a copy of the training label

### Pipeline for numerical variables
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('std_scaler', StandardScaler())])

### ColumnTransformer for training data set
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer 
num_attribs = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
cat_attribs = ["island","sex"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)])

X_train_tr = full_pipeline.fit_transform(training_data)  #the final training data excluding the labels

#### apply same transformation to test data 
test_data = X_test.copy()
test_labels = y_test.copy()

### Pipeline for numerical variables
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('std_scaler', StandardScaler())])

### ColumnTransformer for training data set
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer 
num_attribs = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
cat_attribs = ["island","sex"]

full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs),("cat", OneHotEncoder(), cat_attribs)])

X_test_tr = full_pipeline.fit_transform(test_data) # the final test data , excluding the test labels

## Train and compare ML models
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# use these classifiers, without changing their settings
perceptron_clf = SGDClassifier(max_iter=100, loss='perceptron', random_state=42)
tree_clf = DecisionTreeClassifier(max_features=2, max_depth=2, random_state=42)
ada_clf = AdaBoostClassifier(random_state=42)

from sklearn.model_selection import cross_val_score
CV_score_perceptron = cross_val_score(perceptron_clf, X_train_tr, training_labels, cv=3, scoring="accuracy", verbose=5)
CV_score_tree = cross_val_score(tree_clf, X_train_tr, training_labels, cv=3, scoring="accuracy", verbose=5)
CV_score_ADA = cross_val_score(ada_clf, X_train_tr, training_labels, cv=3, scoring="accuracy", verbose=5)

### codes to generate a plot comparing the CV scores of the classifiers
# combine the name of the classifier and the scores into a dictionary
algo_score = [CV_score_perceptron,CV_score_tree,CV_score_ADA]
algo_names = ['Perceptron','Decision_tree','ADA']
dicts={}
values = []
for score in algo_score:
    mean = round(score.mean(),4)
    std = round(score.std(),4)
    values.append((mean, std))
for i in range(len(algo_names)):
    dicts[algo_names[i]] = values[i]

# convert the dictionary to a pandas dataframe, then generate a plot
df_CV_scores = pd.DataFrame.from_dict(dicts, orient='index', columns=['Mean','Std'])
x = df_CV_scores.index
y = df_CV_scores['Mean']
e = df_CV_scores['Std'] # standard deviation
plt.bar(x,y, color="grey")  # plots the mean
plt.ylabel("Mean")
plt.errorbar(x, y, yerr=e, fmt='o',ecolor = 'teal',color='black',marker = "|") # plots the standard deviation
plt.show()
df_CV_scores # prints the data frame

## Error Analysis
# Use cross validation to generate predictions for plotting the confusion matrix
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(ada_clf, X_train_tr, training_labels, cv=3)

# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(training_labels, y_train_pred)

## Calculate the true positive rate(TPR) and true negative rate (TNR) of the ADA classifier
TP = 57
FP = 1
FN = 2
TN = 237

TPR = round((TP/(TP+FN))*100,2)
print("TPR of the ADA classifier is "+str(TPR)+'%')
TNR = round((TN/(FP+TN))*100,2)
print("TNR of the ADA classifier is "+str(TNR)+'%')

## precision and recall
from sklearn.metrics import precision_score, recall_score
print("precision score of the Perceptron classifier is: ",precision_score(training_labels, y_train_pred)) # accuracy of positive predictions
print("recall score of the Perceptron classifier is: ",recall_score(training_labels, y_train_pred))









