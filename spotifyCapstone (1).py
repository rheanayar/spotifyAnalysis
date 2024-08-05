#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 10:01:00 2024

@author: rheanayar
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import f_oneway

#preprocessing: 
#First I will be checking for null values 
data = pd.read_csv('/Users/rheanayar/Downloads/spotify52kData.csv' )
data.info() #there seems to be no null values, to double check, and explore data furthur I will look for unique values 
data['duration'] = data['duration'] / 60000 # converting miliseconds to minutes

for column in data.columns:
    unique_values = data[column].unique()
    print("Unique values in column" ,column,":" , unique_values) #no null values seen 
data.dropna() #to double confirm 
#after looking at the unique values, I will normalize energy, speechiness, acousticness, instrumentalness, and liveness throughout 

num_bins = 31 #to be used throughout
random_seed = 2357 #for cross validation purposes 

#1.) Consider the 10 songs that feature duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence and tempo. 
#Are any of these features reasonably distributed normally? If so, which one? 
#list of features 
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo' ]


fig, axes = plt.subplots(nrows = 2, ncols = 5, figsize=(12, 4))
#making the 2d array into 1d array 
axes = axes.flatten()

#looping through every feature and creating a histogram 
for i, feature in enumerate(features): 
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].hist(data[feature], bins=num_bins)
    axes[i].set_title(feature)
    
plt.tight_layout() #for better appearance 
plt.show()

#2 Is there a relationship between song length and popularity of a song? If so, if the relationship
#positive or negative? 

#since duration is skewed, Im using log transormation to reveal if there is any relationship 
data['log_duration'] = np.log(data['duration'] + 1)

#cannot decipher any pattern, so I decided to use both to see which one better aligns..
pearson_corr, pearson_p_value = pearsonr(data['log_duration'], data['popularity'])
spearman_corr, spearman_p_value = spearmanr(data['log_duration'], data['popularity'])

#scatter plot for log transformed duration
sns.scatterplot(x=data['log_duration'], y=data['popularity'])
plt.title('Distribution of Log-transformed Duration vs. Popularity')
plt.xlabel('Log-Transformed Duration (minutes)')
plt.ylabel('Popularity')
plt.show()

#printing correlation results 
print("Pearson correlation:" ,pearson_corr," p value:" ,pearson_p_value)
print("Spearman correlation:",spearman_corr,"p value:", spearman_p_value)


#3 Are explicitly rated songs more popular than songs that are not explicit? 

#filtering the data between explicit and non=explciit songs
explicit_songs = data[data['explicit'] == True] # all rows of explicit songs
non_explicit_songs = data[data['explicit'] == False] # all rows of non explicit songs

# summary statistics and plotting distribution for observation
print("explicit songs median", explicit_songs['popularity'].median())
print("not explicit songs median", non_explicit_songs['popularity'].median())

sns.histplot(explicit_songs['popularity'], bins=num_bins, color='red')
plt.xlabel('Popularity Score')
plt.ylabel('Count')
plt.title('Distribution of Popularity Based on Explicit Songs')
plt.show()
sns.histplot(non_explicit_songs['popularity'], bins= num_bins, color='blue')
plt.title('Distribution of Popularity Based on Non-Explicit Songs')
plt.xlabel('Popularity Score')
plt.ylabel('Count')
plt.show()
#h0:  there is no statistically significant difference between non-explicit and explicit songs. 
statistic, p_value = mannwhitneyu(explicit_songs['popularity'], non_explicit_songs['popularity'])
print("p value for U test:", p_value)

#4 Are songs in major key more popular than songs in minor key?
#dataframe of just the mode and popularity 
key_pop = ['mode', 'popularity']
key_pop_data = data[key_pop]

#retrieving their medians
print("major songs median", explicit_songs['popularity'].median())
print("minor songs median", non_explicit_songs['popularity'].median())

# Define major and minor keys according to dataset
major_keys = [1]
minor_keys = [0]

#filtering the data based on key (major or mode)
major_pop = key_pop_data[key_pop_data['mode'].isin(major_keys)]['popularity']
minor_pop = key_pop_data[key_pop_data['mode'].isin(minor_keys)]['popularity']

#plotting both histograms
plt.hist(major_pop, bins=num_bins, color='pink')
plt.xlabel('Popularity')
plt.ylabel('Density')
plt.title('Distribution of Popularity for Major Key')
plt.legend()
plt.show()

plt.hist(minor_pop, bins=num_bins, color='red')
plt.xlabel('Popularity')
plt.ylabel('Density')
plt.title('Distribution of Popularity for Minor Key ')
plt.legend()
plt.show()
#These distirbutions are not normally distributed
#Performing mann-whitney u test 

U_stat_key, p_value_key = mannwhitneyu(minor_pop, major_pop)
print("p value for U test between major and minor songs", p_value_key)

#5 Energy is believed to largely reflect the “loudness” of a song. Can you substantiate (or refute) that this is the case? [Suggestion: Include a scatterplot

#plotting in order to analyze relationship and see which correlation coefficent to use
#looks monotonic, so spearman rho will be used 

scaler = StandardScaler()

# normalize 'loudness' and 'energy' to put on the same scale to help observe overall pattern
data[['loudness_normalized', 'energy_normalized']] = scaler.fit_transform(data[['loudness', 'energy']])

#creating a scatterplot of energy and loudness
plt.scatter(data['loudness_normalized'], data['energy_normalized'])
plt.xlabel('Energy')
plt.ylabel('Loudness')
plt.show()

#monotonic relationship, so I decided to use spearmans rho 
rho, pval_rho = stats.spearmanr(data['energy'],data['loudness'])  

print(rho) #correlation of rho is .7306

#6 Which of the 10 individual (single) song features from question 1 predicts popularity best? How good is this “best” model?

#in order to determine which predicts popularity the best,a simple linear regression can be done 
#figuring out the best score based on R^2,(all the output will be printed) , which will take in y and y predicted as its parameters
#once the best predictor is found, it is normalized and put into a linear regression model 
#scatterplot for visualization 

for feature in features: 
   x = data[[feature]]
   y = data['popularity']
 
   model = LinearRegression()
   model.fit(x, y)
   yHat = model.predict(x)

   
   r_score = r2_score(y, yHat)
   rmse = np.sqrt(mean_squared_error(y, yHat))
    
   print("Feature:" ,feature)
   print("R^2 Score",r_score)
   print("RMSE" ,rmse)
X = data[['instrumentalness']]
y = data['popularity']

#normalizing instrumentalness
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#creating linear regresison model 
model = LinearRegression()
model.fit(X_scaled, y)

#predict and evaluate
y_pred = model.predict(X_scaled)
rmse = np.sqrt(mean_squared_error(y, yHat))
r2 = r2_score(y, y_pred)
print("RMSE for simple regression", rmse)
print("R^2 for simple regression", r2)

#scatterplot
plt.scatter(X_scaled, y, alpha=0.3)
plt.plot(X_scaled, y_pred, color='red')  #regression line
plt.xlabel('Instrumentalness (scaled)')
plt.ylabel('Popularity')
plt.title('Instrumentalness vs. Popularity with Simple Regression')
plt.show()


#7 Building a model that uses *all* of the song features from question 1, how well can you predict popularity now? 
#How much (if at all) is this model improved compared to the best model in question 6). How do you account for this?             

# multiple regression model 
X = data[features]
y = data['popularity']

#initalizing standard scaler , then transforming the data by standardizing it 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#linear regression model
full_model = LinearRegression()
full_model.fit(X_scaled, y)
yHat_full = full_model.predict(X_scaled)

# calculating R^2 as well as RMSE 
r2_full = r2_score(y, yHat_full)
rmse_full = np.sqrt(mean_squared_error(y, yHat_full))
print("RMSE for Multiple Regression", rmse_full)
print("R^2 for Multiple Regression" ,r2_full)

# plotting predictions vs actual values
plt.plot(yHat_full, y, 'o', markersize=4)
plt.xlabel('Prediction from model')
plt.ylabel('Actual popularity')
plt.title("Actual vs. Predicted Popularity")
plt.show()

#8) When considering the 10 song features above, how many meaningful principal components can you extract?
# What proportion of the variance do these principal components account for?

#first lets do EDA by finding correlation matrix
corr = data.corr()
plt.figure(figsize=(12, 10)) #so that everything fits in the matrix visually 
sns.heatmap(corr, annot=True, fmt='.2f', cbar=True)
plt.title('Correlation Matrix of Song Features')
plt.show()
#from this , I observed a positive .77 correlation between energy and loudness, a negative .74 correlation between energy and acousticness 
#as well as a -.61 correlation with loudness and acousticness

#standardizing
scaler = StandardScaler()
zscored_data = scaler.fit_transform(X) # where X = data[features]

#performing pca on the standardized data 
pca = PCA().fit(zscored_data) 

#e values 
eig_values =  pca.explained_variance_
#loadings 
loadings = pca.components_ 

#transforming the data 
principal_components = pca.transform(zscored_data)

#calculating the amount of variance explained by each principal component
var_explained = eig_values/sum(eig_values)*100

# displaying this for each factor:
for component, variance in enumerate(var_explained ):
    print("Principal Component:", component +1, ":", variance, "%")

#creating screeplot 
numFeatures = len(features)
x = np.linspace(1,numFeatures,numFeatures)
plt.bar(x, eig_values, color='gray')
plt.plot([0,numFeatures],[1,1],color='pink') # pink line representing  Kaiser criterion line 
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()  

#9.) Can you predict whether a song is in major or minor key from valence? 
#If so, how good is this prediction? If not, is there a better predictor? [Suggestion: It might be nice to show the logistic regression once you are done building the model]

#calculating correlation between valence and mode
corr_valence = data[['valence', 'mode']].corr() 
print('correlation', corr_valence)
sns.histplot(data = data, x= 'valence', hue = 'mode', bins = num_bins) #using Seaborn to create histogram
plt.title('Valence Distribution by Key')
plt.xlabel('Valence')
plt.ylabel('count')
plt.legend(title='mocd', labels=['Minor', 'Major'])
plt.show()

#conducting logistic regression using valence as predictor
X = data[['valence']]
y = data['mode']

# splitting into training and testing sets (20% for testing, 80% for training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# building and training a logistic regresstion model 
model = LogisticRegression()
model.fit(X_train, y_train)

# making predictions 
y_pred = model.predict(X_test)

# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) #accuracy of the predictions
report = classification_report(y_test, y_pred) #creating a classification report
conf_matrix = confusion_matrix(y_test, y_pred) #calculating the confusion matrix
print("Results from valence" )
print("Accuracy:", accuracy)
print("Classification Report:" ,report)
print("Confustion:", conf_matrix)

#calculating auc
y_pred_proba = model.predict_proba(X_test)[:, 1] #predicting probability of major 

auc = roc_auc_score(y_test, y_pred_proba)
print("AUC:", auc)

#declaring and inisitalizing variables fpr (false positive rate), tpr (true positive rate) , and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
#plotting ROC curve in blue
plt.plot(fpr, tpr, color='blue', label= 'ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') #plotting a diagonal line in order to determine if model is guessing
plt.xlim([0.0, 1.0]) #setting  limit for false positive rate 
plt.ylim([0.0, 1.05]) #setting limit for true positive rate 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right') #adding a legend to add more clarity 
plt.show() #valence is not a good predictor .. close to diagonal line

#trying another predictor variable, key 
X = data[['key']]
y = data['mode']

#splitting data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#predictions
y_pred = model.predict(X_test)

#using AUC to evaluvate model and calculating accuracy, report, and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Results from key" )
print("Accuracy:", accuracy)
print("Classification Report:" ,report)
print("Confustion:", conf_matrix)

y_pred_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
print("AUC:", auc)

#false positive rate, true positive rate, threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve ')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show() #key is a better predictor compared to valence

#trying another better predictor 
X = data[['acousticness']]
y = data['mode']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

#logistic regression 
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#calculating accuracy, report, and the confusion matrix 
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Results from Acousticness" )
print("Accuracy:", accuracy)
print("Classification Report:" ,report)
print("Confustion:", conf_matrix)
y_pred_proba = model.predict_proba(X_test)[:, 1]

#auc being calculated to predict the model 
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC:", auc)

#creating variabled false positive, true positive, and thresholds 
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') #diagonal line 
#limits
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show() #acousticness is a better predictor compared to valence

#10  Which is a better predictor of whether a song is classical music – duration or the principal components you extracted in question 8? 
#[Suggestion: You might have to convert the qualitative genre label to a binary numerical label (classical or not)]

#separating classical and labelling it into a binary numerical label (1 = classical, 0= every other track genre thats not classical)
data['classical'] = data['track_genre'].apply(lambda genre: 1 if genre == 'classical' else 0)
durr_corr = data[['classical', 'duration']].corr()
print('duration correlation', durr_corr) #0.003302
print(data['classical'].value_counts())

#creating logistic regression  model with duration as predictor 
X = data[['duration']]
y= data['classical'] 

#splitting the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:" ,accuracy)
print("Classification Report:" , report)
print("Confusion: "  ,conf_matrix)
y_pred_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
print("AUC", auc)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC  curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
#AUC is .59
#acccuracy is .98

#second model using principal components
y= data['classical'] 

#splitting data 
X_train1, X_test1, y_train1, y_test1 = train_test_split(principal_components, y, test_size=0.2, random_state= random_seed)
classic_model = LogisticRegression()
classic_model.fit(X_train1, y_train1)
y_pred1 = classic_model.predict(X_test1)


#getting accuracy scores, classification report, and the confusion matrix 
accuracy = accuracy_score(y_test1, y_pred1)
report = classification_report(y_test1, y_pred1)
confusion = confusion_matrix(y_test1, y_pred1)

print("Accuracy:", accuracy)
print("Classification Report:",report)
print("Confusion Matrix: " ,confusion)
y_pred_proba1 = classic_model.predict_proba(X_test1)[:, 1]
auc_classic = roc_auc_score(y_test1, y_pred_proba1)
print("AUC for classification:",auc_classic)
fpr, tpr, thresholds = roc_curve(y_test1, y_pred_proba1)

#plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve ')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

#dataset should be split into 5 folds ; shuffle set to true to prevent bias 
kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

# Performing cross validation with the logistic regression model using principal components
cv_scores = cross_val_score(classic_model, principal_components, y, cv=kf, scoring='accuracy')

# Print cross validation scores
print("cross validation scores:", cv_scores)
print("Mean accuracy:",  cv_scores.mean())

#11.) extra cred 
#counting the occurances of each key 
key_counts = data['key'].value_counts()
print("Key Counts:")
print(key_counts)

#calculating the mean dancability for each key , as danceability was roughly normally distributed
mean_popularity_by_key = data.groupby('key')['danceability'].mean()
print("Mean Danceability by Key:")
print(mean_popularity_by_key)

#creating bar plot to visualize the mean danceability per key 
sns.barplot(x=mean_popularity_by_key.index, y=mean_popularity_by_key.values)
plt.title('Mean Danceability by Key')
plt.xlabel('Key')
plt.ylabel('Mean Danceability')
plt.show()

#getting all the unique keys from the dataset( ranging from 0-11)
keys = data['key'].unique()
#grouping all the data by key on danceability  
grouped_data = [data[data['key'] == k]['danceability'] for k in keys]
#perfoming ANOVA test to see if theres significant difference in danceabilitity based on key 
f_statistic, p_value_anova = f_oneway(*grouped_data)
print("ANOVA F statistic:", f_statistic)
print("ANOVA p value:", p_value_anova)
#Statsitically significant results. There is a difference based on keys and danceability. 
