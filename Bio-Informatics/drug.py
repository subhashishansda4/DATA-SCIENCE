# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:50:12 2022

@author: VAGUE
"""

'''
"John S. Delaney" - ESOL: Estimating Aqueous Solubility Directly from Molecular Structure (research paper)
linear regression model for predicting molecular solubility

"Pat Walters" - Deep Learning for the Life Sciences (author)
'''

# DATA COLLECTION
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style='ticks')
import matplotlib.pyplot as plt
# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# machine learning
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# rdkit for lipinski descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
# CHEMBL database
from chembl_webresource_client.new_client import new_client

# search for target protein
target = new_client.target
target_query = target.search('acetylcholinesterase')
targets = pd.DataFrame.from_dict(target_query)
print('Targets Dataframe')
print(targets)
print('')

# select and retreive bioactivity data
selected_target = targets.target_chembl_id[0]
print('Selected Target:', selected_target)
print('')

# filtering by selected_target and standard_type
activity = new_client.activity
res = activity.filter(target_chembl_id = selected_target).filter(standard_type = 'IC50')

# creating dataframe
df = pd.DataFrame.from_dict(res)
print(df.head(3))
print('')
print(df.standard_type.unique())
print('')

# dataframe to csv
df.to_csv('data/bioactivity_data_raw.csv', index = False)

# dropping missing value for the standard_value and canonical_smiles column
df = df[df.standard_value.notna()]
df = df[df.canonical_smiles.notna()]
print('total unique smiles notations', len(df.canonical_smiles.unique()))
# dropping duplicate canonical_smiles
df = df.drop_duplicates(['canonical_smiles'])
print(df)
print('')

# data pre-processing
# labelling compound as either being active, inactive or intermediate
bioactivity_class = []
for i in df.standard_value:
    if float(i) >= 10000:
        bioactivity_class.append('inactive')
    elif float(i) <= 1000:
        bioactivity_class.append('active')
    else:
        bioactivity_class.append('intermediate')

# iterating through molecule_chembl_id
mol_cid = []
for i in df.molecule_chembl_id:
    mol_cid.append(i)
    
# iterating through canonical_smiles
canonical_smiles = []
for i in df.canonical_smiles:
    canonical_smiles.append(i)
    
# iterating through standard_value
standard_value = []
for i in df.standard_value:
    standard_value.append(i)

# combining the 4 lists into a dataframe
data_tuples = list(zip(mol_cid, canonical_smiles, bioactivity_class, standard_value))
df_bioclass = pd.DataFrame(data_tuples, columns=['molecule_chembl_id', 'canonical_smiles', 'bioactivity_class', 'standard_value'])
        
# =============================================================================
# # selecting different parameters to iterate through and make them unique
# selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
# df_filter = df[selection]
# # merging dataframes
# df_bioclass = pd.concat([df_filter, pd.Series(bioactivity_class)], axis=1)
# =============================================================================

# dataframe to csv
df_bioclass.to_csv('data/bioactivity_preprocessed_data.csv', index = False)
print('Preprocessed Dataframe')
print(df_bioclass)
print('')
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# DATA CLEANING & PROCESSING
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
df_no_smiles = df_bioclass.drop(columns='canonical_smiles')

smiles = []
for i in df_bioclass.canonical_smiles.tolist():
  cpd = str(i).split('.')
  cpd_longest = max(cpd, key = len)
  smiles.append(cpd_longest)
  
smiles = pd.Series(smiles, name = 'canonical_smiles')
df_clean_smiles = pd.concat([df_no_smiles,smiles], axis=1)
print('Clean Dataframe')
print(df_clean_smiles)
print('')

# source - https://codeocean.com/explore/capsules?query=tag:data-curation
# calculating lipinski descriptors
def lipinski(smiles, verbose = False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)
        
    baseData = np.arange(1,1)
    i=0
    for mol in moldata:
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
        
        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors
                    ])
        
        if(i==0):
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i=i+1
    
    columnNames = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)
    
    return descriptors

# dataframe
df_lipinski = lipinski(df_clean_smiles.canonical_smiles)
print('Lipinski Descriptors')
print(df_lipinski)
print('')
df_lipinski = pd.concat([df_clean_smiles, df_lipinski], axis=1)

# dataframe to csv
df_lipinski.to_csv('data/lipinski_descriptors.csv', index=False)

# NOTE
# values greater than 100,000,000 will be fixed at that value
# otherwise the negative logarithmic value will become negative
print(df_lipinski.standard_value.describe())
print('')
print(-np.log10((10**-9)*100000000))
print(-np.log10((10**-9)*10000000000))

# changing data type of 'standard_value'
df_lipinski['standard_value'] = df_lipinski['standard_value'].astype(float)
print(df_lipinski.dtypes)
print('')

# capping the standard_value
def norm_value(input):
    norm = []
    
    for i in input['standard_value']:
        if i > 100000000:
            i = 100000000
        norm.append(i)
    
    input['standard_value_norm'] = norm
    x = input.drop('standard_value', 1)
    x = input
    
    return x


# converting IC50 to pIC50
# source - https://github.com/chaninlab/estrogen-receptor-alpha-qsar/blob/master/02_ER_alpha_RO5.ipynb
def pIC50(input):
    pIC50 = []
    
    for i in input['standard_value_norm']:
        molar = float(i)*(10**-9)  # converts nM to M
        pIC50.append(-np.log10(molar))
        
    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', 1)
    
    return x

# applying normalization of values
df_final = norm_value(df_lipinski)
# applying conversion of IC50 to pIC50
df_final = pIC50(df_final)
# statistics of standard_value and pIC50 values
print(df_final.standard_value.describe())
print('')
print(df_final.pIC50.describe())
print('')
# removing 'intermediate' bioactivity class
df_final = df_final[df_final.bioactivity_class != 'intermediate']
print('Final Dataframe')
print(df_final)

# dataframe to csv
df_final.to_csv('data/final_data.csv', index=False)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# EDA (CHEMICAL SPACE ANALYSIS)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
Jose Medina Franco (author)
each chemical compound could be thought of as stars, i.e. active molecules would be compared to as constellations
he developed an approach termed as "Constellation Plot" whereby one can perform chemical space analysis & create constellation plot where

[active molecule would be correspondingly have larger sizes compared to less active molecule]
'''
# frequency plot of bioactivity classes comparing inactive and active molecules
plt.figure(figsize = (5.5, 5.5))
sns.countplot(x = 'bioactivity_class', data = df_final, edgecolor = 'black')

plt.xlabel('Bioactivity class', fontsize = 14, fontweight = 'bold')
plt.ylabel('Frequency', fontsize = 14, fontweight = 'bold')

plt.savefig('plots/plot_bioactivity_class.jpg')

# scatter plot of molecular weight(MW) v/s molecular solubility(logP)
plt.figure(figsize = (5.5, 5.5))
sns.scatterplot(x = 'MW', y = 'LogP', data = df_final, hue = 'bioactivity_class', size = 'pIC50', edgecolor = 'black', alpha = 0.7)

plt.xlabel('MW', fontsize = 14, fontweight = 'bold')
plt.ylabel('LogP', fontsize = 14, fontweight = 'bold')
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0)

plt.savefig('plots/plot_MW_vs_logP.jpg')

# scatter plot of IC50 vs pIC50
plt.figure(figsize = (5.5, 5.5))
sns.scatterplot(x = 'standard_value', y = 'pIC50', data = df_final, hue = 'bioactivity_class', edgecolor = 'black', alpha = 0.7)

plt.xlabel('IC50', fontsize = 14, fontweight = 'bold')
plt.ylabel('pIC50', fontsize = 14, fontweight = 'bold')
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0)

plt.savefig('plots/IC50_vs_pIC50.jpg')
# mann-whitney U test
# source - https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python
def mannwhitney(descriptor, verbose = False):
    from numpy.random import seed
    from scipy.stats import mannwhitneyu
    
    # seeding the random number generator
    seed(1)
    
    # actives and inactives
    selection = [descriptor, 'bioactivity_class']
    df = df_final[selection]
    active = df[df.bioactivity_class == 'active']
    active = active[descriptor]
    
    selection = [descriptor, 'bioactivity_class']
    df = df_final[selection]
    inactive = df[df.bioactivity_class == 'inactive']
    inactive = inactive[descriptor]
    
    # compare samples
    stat, p = mannwhitneyu(active, inactive)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    
    # interpret
    alpha = 0.05
    if p > alpha:
        interpretation = 'Same distribution (fail to reject H0)'
    else:
        interpretation = 'Different distribution (reject H0)'
        
    results = pd.DataFrame({'Descriptor': descriptor,
                            'Statistics': stat,
                            'p': p,
                            'alpha': alpha,
                            'Interpretation': interpretation}, index=[0])
    
    filename = 'mannwhitneyu_' + descriptor + '.csv'
    results.to_csv('plots/' + filename)
    
    return results

# box plots for pIC50, MW, LogP, NumHDonors, NumHAcceptors
# performing mann-whitney analysis for each one of them

# pIC50
plt.figure(figsize = (5.5, 5.5))
sns.boxplot(x = 'bioactivity_class', y = 'pIC50', data = df_final)
plt.xlabel('Bioactivity_class', fontsize = 14, fontweight = 'bold')
plt.ylabel('pIC50 value', fontsize = 14, fontweight = 'bold')
plt.savefig('plots/plot_ic50.jpg')
mannwhitney('pIC50')

# MW
plt.figure(figsize = (5.5, 5.5))
sns.boxplot(x = 'bioactivity_class', y = 'MW', data = df_final)
plt.xlabel('Bioactivity_class', fontsize = 14, fontweight = 'bold')
plt.ylabel('MW', fontsize = 14, fontweight = 'bold')
plt.savefig('plots/plot_MW.jpg')
mannwhitney('MW')

# LogP
plt.figure(figsize = (5.5, 5.5))
sns.boxplot(x = 'bioactivity_class', y = 'LogP', data = df_final)
plt.xlabel('Bioactivity_class', fontsize = 14, fontweight = 'bold')
plt.ylabel('LogP', fontsize = 14, fontweight = 'bold')
plt.savefig('plots/plot_LogP.jpg')
mannwhitney('LogP')

# NumHDonors
plt.figure(figsize = (5.5, 5.5))
sns.boxplot(x = 'bioactivity_class', y = 'NumHDonors', data = df_final)
plt.xlabel('Bioactivity_class', fontsize = 14, fontweight = 'bold')
plt.ylabel('NumHDonors', fontsize = 14, fontweight = 'bold')
plt.savefig('plots/plot_NumHDonors.jpg')
mannwhitney('NumHDonors')

# NumHAcceptors
plt.figure(figsize = (5.5, 5.5))
sns.boxplot(x = 'bioactivity_class', y = 'NumHAcceptors', data = df_final)
plt.xlabel('Bioactivity_class', fontsize = 14, fontweight = 'bold')
plt.ylabel('NumHAcceptors', fontsize = 14, fontweight = 'bold')
plt.savefig('plots/NumHAcceptors.jpg')
mannwhitney('NumHAcceptors')

print('')
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# DATA PREPARATION
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
selection = ['canonical_smiles', 'molecule_chembl_id']
df_final_selection = df_final[selection]
df_final_selection.to_csv('padel/molecule.smi', sep='\t', index=False, header=False)

# preparing X and Y matrices
# X
df_final_X = pd.read_csv('padel/descriptors_output.csv')
df_final_X = df_final_X.drop(columns=['Name'])

df_drop = df_final.drop(columns=['molecule_chembl_id', 'bioactivity_class', 'standard_value', 'canonical_smiles', 'pIC50'])
df_drop = df_drop.reset_index()
df_drop = df_drop.drop(columns=['index'])
df_final_X = pd.concat([df_final_X, df_drop], axis=1, ignore_index=True)

# Y
df_final_Y = df_final['pIC50']
df_final_Y = df_final_Y.reset_index()
df_final_Y = df_final_Y.drop(columns=['index'])

# combining X and Y variable
dataset = pd.concat([df_final_X, df_final_Y], axis=1)

# missing values
missing_values = dataset.isnull().sum()
missing_values[0:886]

total_cells = np.product(dataset.shape)
total_missing = missing_values.sum()
percent_missing = (total_missing/total_cells)*100
print(percent_missing)
print('')

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
dataset = clean_dataset(dataset)

# dataframe to csv
dataset.to_csv('data/dataset.csv', index=False)
print('PubChem Fingerprints')
print(dataset)
print('')

# Principal Component Analysis
# source - https://www.kaggle.com/code/ankitjha/comparing-regression-models/notebook
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# normalization
#data = StandardScaler().fit_transform(dataset)
data = dataset

pca = PCA().fit(data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,60,1)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative explained variance')
plt.savefig('plots/PCA.jpg')
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# SPLITTING DATA
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
# input features
data_X = data.drop('pIC50', axis=1)
# using number of components to be 80
# since it can explain almost 40% of the variance
X = PCA(n_components=40).fit_transform(data_X)
# output features
Y = data.pIC50
# data dimension
print('X=', X.shape)
print('Y=', Y.shape)
print('')
'''

X = data.drop('pIC50', axis=1)
Y = data.pIC50

# remove low variance features
selection = VarianceThreshold(threshold=(.8*(1-.8)))
X = selection.fit_transform(X)
print('X=', X.shape)
print('')


# data split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print('shapes:')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print('')

Y = Y.to_frame()
y_test = y_test.to_frame()
y_train = y_train.to_frame()
y_test = y_test.values.ravel()
y_train = y_train.values.ravel()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# MACHINE LEARNING MODELS
# source - https://dibyendudeb.com/comparing-machine-learning-regression-models-using-python/
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MULTIPLE LINEAR REGRESSION
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
# predicting using test set
y_pred1 = lin_reg.predict(x_test)
# mean absolute error
mae1 = metrics.mean_absolute_error(y_test, y_pred1)
# mean square error
mse1 = metrics.mean_squared_error(y_test, y_pred1)
# r2 square
r21 = metrics.r2_score(y_test, y_pred1)
# scores
print('Multiple Linear Regression')
print(mae1)
print(mse1)
print(r21)
print('')

# scatter plot of experimental vs predicted pIC50 values
plt.scatter(y_test, y_pred1)
plt.xlabel('Experimental pIC50')
plt.ylabel('Predicted pIC50')
lims = [0,12]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig('plots/Multiple Linear Regression.jpg')


# DECISION TREE REGRESSION
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x_train, y_train)
# predicting using test set
y_pred2 = dt_reg.predict(x_test)
# mean absolute error
mae2 = metrics.mean_absolute_error(y_test, y_pred2)
# mean square error
mse2 = metrics.mean_squared_error(y_test, y_pred2)
# r2 square
r22 = metrics.r2_score(y_test, y_pred2)
# scores
print('Decision Tree Regression')
print(mae2)
print(mse2)
print(r22)
print('')

# scatter plot of experimental vs predicted pIC50 values
plt.scatter(y_test, y_pred2)
plt.xlabel('Experimental pIC50')
plt.ylabel('Predicted pIC50')
lims = [0,12]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig('plots/Decision Tree Regression.jpg')


# RANDOM FOREST REGRESSION
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(x_train, y_train)
# predicting using test set
y_pred3 = rf_reg.predict(x_test)
# mean absolute error
mae3 = metrics.mean_absolute_error(y_test, y_pred3)
# mean square error
mse3 = metrics.mean_squared_error(y_test, y_pred3)
# r2 square
r23 = metrics.r2_score(y_test, y_pred3)
# scores
print('Random Forest Regression')
print(mae3)
print(mse3)
print(r23)
print('')

# scatter plot of experimental vs predicted pIC50 values
plt.scatter(y_test, y_pred3)
plt.xlabel('Experimental pIC50')
plt.ylabel('Predicted pIC50')
lims = [0,12]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig('plots/Random Forest Regression.jpg')


# RIDGE REGRESSION
rdg_reg = Ridge()
rdg_reg.fit(x_train, y_train)
# predicting using test set
y_pred4 = rdg_reg.predict(x_test)
# mean absolute error
mae4 = metrics.mean_absolute_error(y_test, y_pred4)
# mean square error
mse4 = metrics.mean_squared_error(y_test, y_pred4)
# r2 square
r24 = metrics.r2_score(y_test, y_pred4)
# scores
print('Ridge Regression')
print(mae4)
print(mse4)
print(r24)
print('')

# scatter plot of experimental vs predicted pIC50 values
plt.scatter(y_test, y_pred4)
plt.xlabel('Experimental pIC50')
plt.ylabel('Predicted pIC50')
lims = [0,12]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig('plots/Ridge Regression.jpg')


# BAYESIAN REGRESSION
by_reg = BayesianRidge()
by_reg.fit(x_train, y_train)
# predicting using test set
y_pred5 = by_reg.predict(x_test)
# mean absolute error
mae5 = metrics.mean_absolute_error(y_test, y_pred5)
# mean square error
mse5 = metrics.mean_squared_error(y_test, y_pred5)
# r2 square
r25 = metrics.r2_score(y_test, y_pred5)
# scores
print('Bayesian Regression')
print(mae5)
print(mse5)
print(r25)
print('')

# scatter plot of experimental vs predicted pIC50 values
plt.scatter(y_test, y_pred5)
plt.xlabel('Experimental pIC50')
plt.ylabel('Predicted pIC50')
lims = [0,12]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig('plots/Bayesian Regression.jpg')


# K-NEAREST NEIGHBOUR
n = 5
knn_reg = KNeighborsRegressor(n, weights='uniform')
knn_reg.fit(x_train, y_train)
# predicting using test set
y_pred6 = knn_reg.predict(x_test)
# mean absolute error
mae6 = metrics.mean_absolute_error(y_test, y_pred6)
# mean square error
mse6 = metrics.mean_squared_error(y_test, y_pred6)
# r2 square
r26 = metrics.r2_score(y_test, y_pred6)
# scores
print('K-Nearest Neighbour')
print(mae6)
print(mse6)
print(r26)
print('')

# scatter plot of experimental vs predicted pIC50 values
plt.scatter(y_test, y_pred6)
plt.xlabel('Experimental pIC50')
plt.ylabel('Predicted pIC50')
lims = [0,12]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig('plots/K-Nearest Neighbour.jpg')


# SUPPORT VECTOR REGRESSION
sv_reg = SVR(kernel='rbf')
sv_reg.fit(x_train, y_train)
# predicting using test set
y_pred7 = sv_reg.predict(x_test)
# mean absolute error
mae7 = metrics.mean_absolute_error(y_test, y_pred7)
# mean square error
mse7 = metrics.mean_squared_error(y_test, y_pred7)
# r2 square
r27 = metrics.r2_score(y_test, y_pred7)
# scores
print('Support Vector Regression')
print(mae7)
print(mse7)
print(r27)
print('')

# scatter plot of experimental vs predicted pIC50 values
plt.scatter(y_test, y_pred7)
plt.xlabel('Experimental pIC50')
plt.ylabel('Predicted pIC50')
lims = [0,12]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig('plots/Support Vector Regression.jpg')


# XGBOOST REGRESSION
xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=0)
xgb_reg.fit(x_train, y_train)
# predicting using test set
y_pred8 = xgb_reg.predict(x_test)
# mean absolute error
mae8 = metrics.mean_absolute_error(y_test, y_pred8)
# mean square error
mse8 = metrics.mean_squared_error(y_test, y_pred8)
# r2 square
r28 = metrics.r2_score(y_test, y_pred8)
# scores
print('XGBoost Regressor')
print(mae8)
print(mse8)
print(r28)
print('')

# scatter plot of experimental vs predicted pIC50 values
plt.scatter(y_test, y_pred8)
plt.xlabel('Experimental pIC50')
plt.ylabel('Predicted pIC50')
lims = [0,12]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig('plots/XGBoost Regression.jpg')


# DATA VISUALIZATION
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
models = ['Multiple Linear Regression', 'Decision Tree Regression', 'Random Forest Regression', 'Ridge Regression', 'Bayesian Regression', 'K-Nearest Neighbour', 'Support Vector Regression', 'XGBoost Regression']
mae_values = [mae1, mae2, mae3, mae4, mae5, mae6, mae7, mae8]
mse_values = [mse1, mse2, mse3, mse4, mse5, mse6, mse7, mse8]
r2_values = [r21, r22, r23, r24, r25, r26, r27, r28]

bar = pd.DataFrame(list(zip(models, mae_values, mse_values, r2_values)), columns=['Models', 'Absolute', 'Square', 'R2'])

# figure

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# CONCLUSION
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(x_train, y_train)
# predicting using test set
y_pred = rf_reg.predict(x_test)

output = pd.DataFrame({'pIC50':y_test, 'Predictions':y_pred})
print(output)
print(output.describe())

# dataframe to csv
output.to_csv('data/Predictions.csv', index=False)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------