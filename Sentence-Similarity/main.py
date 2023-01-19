# libraries
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default="browser"
'''pio.renderers.default="svg"'''

import re
from bs4 import BeautifulSoup
import distance

# data import
df = pd.read_csv('data/train.csv')
df.info()

# -------------------------------------------------
# missing and duplicate values
print('')
print('Missing Values')
print(df.isnull().sum())
print('')
print('Duplicate Rows')
print(df.duplicated().sum())

df.dropna(axis=0, inplace=True)

# check
print('')
print('Missing Values')
print(df.isnull().sum())
print('')
print('Duplicate Rows')
print(df.duplicated().sum())

# distribution of duplicate and non-duplicate questions
print('')
print(df['is_duplicate'].value_counts())
print((df['is_duplicate'].value_counts()/df['is_duplicate'].count())*100)
df['is_duplicate'].value_counts().plot(kind='bar').figure.savefig('plots/similar_vs_unique.jpg')

# repeated questions
print('')
print('Repeated Questions')
qid = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
print('Number of unique questions', np.unique(qid).shape[0])
x = qid.value_counts()>1
print('Number of questions getting repeated', x[x].shape[0])
# histogram
print('')
plt.hist(qid.value_counts().values, bins=160)
plt.yscale('log')
plt.show()
plt.savefig('plots/repeat_number.jpg')
# --------------------------------------------------


# --------------------------------------------------
# sample dataset
sample_size = 404287
df_new = df.sample(sample_size, random_state=0)
# --------------------------------------------------


#def process_batch(x):
# --------------------------------------------------
# data preprocessing
# lemmatization and speech tagging
import spacy
nlp = spacy.load("en_core_web_sm")

# token lemmatized speech
def tls(sen):
    doc = nlp(sen)
    speech = [token.lemma_ for token in doc]
    return speech

from words import sym
from words import contractions

def preprocess(q):
    #q = str(q).lower().strip()
    
    for i in range(len(sym)):
        words_ = [word.replace(sym[i], "") for word in q]
    q = ''.join(words_)
    
    # replace certain special characters with their string equivalents
    q = q.replace('%', ' percent ')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    # '[math]' appears around 900 times in the whole dataset
    q = q.replace('[math]', '')
    
    # replacing some numbers with string equivalents
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)
        
    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # removing html tags
    q = BeautifulSoup(q, features="html.parser")
    q = q.get_text()
    
    # remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()
    
    # using tls
    words__ = tls(q)
    q = ' '.join(words__)
    
    
    return q


df_new['question1'] = df_new['question1'].apply(preprocess)
df_new['question2'] = df_new['question2'].apply(preprocess)
# ----------------------------------------------------


# --------------------------------------------------
# feature engineering
# length of string
df_new['q1_len'] = df_new['question1'].str.len()
df_new['q2_len'] = df_new['question2'].str.len()

# number of words
df_new['q1_num_words'] = df_new['question1'].apply(lambda row: len(row.split(" ")))
df_new['q2_num_words'] = df_new['question2'].apply(lambda row: len(row.split(" ")))

# common words
def common_words(row):
    # unique words
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2)
df_new['word_common'] = df_new.apply(common_words, axis=1)

# total common words
def total_common(row):
    # unique words
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    '''return (len(w1) + len(w2))'''
    return (len(w1) + len(w2)) - (len(w1 & w2))
df_new['common_total'] = df_new.apply(total_common, axis=1)

# total words
def total_words(row):
    # unique words
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return (len(w1) + len(w2))
df_new['word_total'] = df_new.apply(total_words, axis=1)

# words shared
df_new['word_share'] = round(df_new['word_common']/df_new['word_total'], 2)
# --------------------------------------------------


# --------------------------------------------------
# advanced features
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')

# token features
def fetch_token_features(row):
    q1 = row['question1']
    q2 = row['question2']
    
    token_features = [0.0]*8
    SAFE_DIV = 0.0001
    STOP_WORDS = stopwords.words("english")
    
    # converting the sentence into tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    
    # get the non-stopwords in questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    # get the stopwords in questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # get the common non-stopwords from question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # get the common stopwords from question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # get the common tokens from question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    # min max
    # cwc min
    # (num of common words) / min(words in q1,q2)
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    # cwc max
    # (num of common words) / max(words in q1,q2)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    # csc min
    # (num of common stop words) / min(stop words in q1,q2)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    # csc max
    # (num of common stop words) / max(stop words in q1,q2)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    # ctc min
    # (num of common tokens) / min(tokens in q1,q2)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    # ctc max
    # (num of common tokens) / max(tokens in q1,q2)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # first word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features

# applying token features
token_features = df_new.apply(fetch_token_features, axis = 1)

df_new['cwc_min'] = list(map(lambda x: x[0], token_features))
df_new['cwc_max'] = list(map(lambda x: x[1], token_features))
df_new['csc_min'] = list(map(lambda x: x[2], token_features))
df_new['csc_max'] = list(map(lambda x: x[3], token_features))
df_new['ctc_min'] = list(map(lambda x: x[4], token_features))
df_new['ctc_max'] = list(map(lambda x: x[5], token_features))
df_new['last_word_eq'] = list(map(lambda x: x[6], token_features))
df_new['first_word_eq'] = list(map(lambda x: x[7], token_features))



# length based features
def fetch_length_features(row):
    q1 = row['question1']
    q2 = row['question2']
    
    length_features = [0.0]*3
    
    # converting the sentence into tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    # absolute length difference
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    # average token length of both questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    
    # longest substring ratio
    strs = list(distance.lcsubstrings(q1, q2))
    if strs:    
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    else:
        length_features[2] = 0
    
    return length_features
    
# applying length features
length_features = df_new.apply(fetch_length_features, axis = 1)

df_new['abs_len_diff'] = list(map(lambda x: x[0], length_features))
df_new['mean_len'] = list(map(lambda x: x[1], length_features))
df_new['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))



from fuzzywuzzy import fuzz    
def fetch_fuzzy_features(row):
    q1 = row['question1']
    q2 = row['question2']
    
    fuzzy_features = [0.0]*4
    
    # fuzzy ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    
    # fuzzy partial ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    
    # token sort ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    
    # token set ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    
    return fuzzy_features

# applying fuzzy features
fuzzy_features = df_new.apply(fetch_fuzzy_features, axis = 1)

df_new['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
df_new['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
df_new['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
df_new['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))
# --------------------------------------------------


# --------------------------------------------------
ques_df = df_new[['question1', 'question2']]
final_df = df_new.drop(columns=['id', 'qid1', 'qid2', 'question1', 'question2'])
# --------------------------------------------------



# --------------------------------------------------
# vectorization of words
'''from sklearn.feature_extraction.text import CountVectorizer'''
from sklearn.feature_extraction.text import TfidfVectorizer
# merge texts
questions = list(ques_df['question1']) + list(ques_df['question2'])
words = [q.split() for q in questions]

# implementing word2vec
from gensim.models import Word2Vec
'''from words import words_0'''

w2v = Word2Vec(words, vector_size=100, window=5, min_count=1, workers=4)
#w2v.build_vocab(words, update=False)
w2v.train(words, total_examples=len(questions), epochs=10)
w2v.save("pickle/word2vec.model")

# vectorizer model
'''cv = CountVectorizer(max_features=3000)'''
tf_idf = TfidfVectorizer(max_features=300)

# using word2vec model to transform sentences
#questions = [q.split() for q in questions]
questions_t = []
for question in questions:
    q_t = []
    for word in question:
        try:
            q_t.append(w2v.wv[word])
        except Exception:
            pass
    questions_t.append(q_t)
    
sample = []
for arr in questions_t:
    s = ", ".join(str(x) for x in arr)
    s = s.replace("[", "").replace("]", "")
    sample.append(s)


q1_arr, q2_arr = np.vsplit(tf_idf.fit_transform(sample).toarray(), 2)

# converting to dataframe and concatenating
temp_df1 = pd.DataFrame(q1_arr, index = ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index = ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis = 1)

# concatenating final_df and ques_df
final_df = pd.concat([final_df, temp_df], axis=1)
#final_df = final_df.drop(columns=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])
    #x = final_df
    # ---------------------------------------------------
    #return x


'''
import concurrent.futures
# input data division into batches
batch_size = 5
batches = [df_new[i:i+batch_size] for i in range(0, len(df_new), batch_size)]

# ThreadPoolExecutor object to run the tasks in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    # tasks submit to executor
    tasks = [executor.submit(process_batch, batch) for batch in batches]
    
    # iterate over completed tasks and get results
    results = [task.result() for task in concurrent.futures.as_completed(tasks)]

final_df = sum(results)
'''

# --------------------------------------------------
# EDA of features
# number of characters
# question 1
print("q1_len")
sns.displot(df_new['q1_len']).savefig('plots/q1_len.jpg')
print('minimum characters', df_new['q1_len'].min())
print('maximum characters', df_new['q1_len'].max())
print('average num of characters', int(df_new['q1_len'].mean()))
print("")
# question 2
print("q2_len")
sns.displot(df_new['q2_len']).savefig('plots/q2_len.jpg')
print('minimum characters', df_new['q2_len'].min())
print('maximum characters', df_new['q2_len'].max())
print('average num of characters', int(df_new['q2_len'].mean()))
print("")

# number of words
# question 1
sns.displot(df_new['q1_num_words']).savefig('plots/q1_num_words.jpg')
print('minimum words', df_new['q1_num_words'].min())
print('maximum words', df_new['q1_num_words'].max())
print('average num of words', int(df_new['q1_num_words'].mean()))
# question 2
sns.displot(df_new['q2_num_words']).savefig('plots/q2_num_words.jpg')
print('minimum words', df_new['q2_num_words'].min())
print('maximum words', df_new['q2_num_words'].max())
print('average num of words', int(df_new['q2_num_words'].mean()))

# common words
sns.displot(df_new[df_new['is_duplicate'] == 0]['word_common'], label='no').savefig('plots/unique_word_common.jpg')
plt.legend()
sns.displot(df_new[df_new['is_duplicate'] == 1]['word_common'], label='yes').savefig('plots/similar_word_common.jpg')
plt.legend()

# total words
sns.displot(df_new[df_new['is_duplicate'] == 0]['word_total'], label='no').savefig('plots/unique_word_total.jpg')
plt.legend()
sns.displot(df_new[df_new['is_duplicate'] == 1]['word_total'], label='yes').savefig('plots/similar_word_total.jpg')
plt.legend()

# total common
sns.displot(df_new[df_new['is_duplicate'] == 0]['common_total'], label='no').savefig('plots/unique_common_total.jpg')
plt.legend()
sns.displot(df_new[df_new['is_duplicate'] == 1]['common_total'], label='yes').savefig('plots/similar_common_total.jpg')
plt.legend()

# words shared
sns.displot(df_new[df_new['is_duplicate'] == 0]['word_share'], label='no').savefig('plots/unique_word_share.jpg')
plt.legend()
sns.displot(df_new[df_new['is_duplicate'] == 1]['word_share'], label='yes').savefig('plots/similar_word_share.jpg')
plt.legend()
# --------------------------------------------------


# --------------------------------------------------
# differentiation plots between various advanced features
sns.pairplot(df_new[['ctc_min', 'cwc_min', 'csc_min', 'is_duplicate']], hue = 'is_duplicate').savefig('plots/token_min.jpg')
sns.pairplot(df_new[['ctc_max', 'cwc_max', 'csc_max', 'is_duplicate']], hue = 'is_duplicate').savefig('plots/token_max.jpg')
sns.pairplot(df_new[['last_word_eq', 'first_word_eq', 'is_duplicate']], hue = 'is_duplicate').savefig('plots/word_order.jpg')
sns.pairplot(df_new[['mean_len', 'abs_len_diff', 'longest_substr_ratio', 'is_duplicate']], hue = 'is_duplicate').savefig('plots/word_stat.jpg')
sns.pairplot(df_new[['fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio', 'is_duplicate']], hue = 'is_duplicate').savefig('plots/fuzz.jpg')
# --------------------------------------------------



# --------------------------------------------------
# t-distributed stochastic neighbour embedding t-SNE
# dimensionality reduction for 15 features to 3 features
'''from sklearn.preprocessing import MinMaxScaler
x = MinMaxScaler().fit_transform(df_new[['cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max', 'last_word_eq', 'first_word_eq', 'mean_len', 'abs_len_diff', 'longest_substr_ratio', 'fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio']])
y = df_new['is_duplicate'].values


from sklearn.manifold import TSNE
tsne3d = TSNE(
    n_components = 3,
    init = 'random', #pca
    random_state = 101,
    method = 'barnes_hut',
    n_iter = 1000,
    verbose = 2,
    angle = 0.5
).fit_transform(x)


# 2d plot for t-SNE
df_new['tsne2d_one'] = tsne2d[:,0]
df_new['tsne2d_two'] = tsne2d[:,1]

sns.scatterplot(
    x = 'tsne2d_one', y = 'tsne2d_two',
    hue = y,
    data = df_new,
    legend = 'full',
    alpha = 0.3
)

# 3d plot for t-SNE
tsne3d_one = tsne3d[:,0]
tsne3d_two = tsne3d[:,1]
tsne3d_three = tsne3d[:,2]

px.scatter_3d(
    df_new,
    x = tsne3d_one, y = tsne3d_two, z = tsne3d_three,
    color=('is_duplicate')
)'''




# ---------------------------------------------------
# train test split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
x_train, x_test, y_train, y_test = train_test_split(final_df.iloc[:,0:-1].values,
                                                    final_df.iloc[:,0].values, test_size=0.3, random_state=0)

'''scores = cross_val_score(pipeline, x_train, y_train, cv=5)'''
kfold = KFold(n_splits=5, shuffle=True, random_state=0)


'''# sentiment analysis
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
y_train = label.fit_transform(y_train)
y_test = label.fit_transform(y_test)'''


from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV

n = 5
# models
lr = LogisticRegression()  
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
knn = KNeighborsClassifier(n)
lda = LinearDiscriminantAnalysis()
svc = SVC(probability=True)
xgb = XGBClassifier()
nb = GaussianNB()
models = [dt, rf, knn, lda, svc, xgb, nb]


# hyperparameter grid
param_grid = {
    "sgdc__loss": [0.1, 1, 10, 100],
    "sgdc__penalty": ['linear', 'poly', 'rbf', 'sigmoid'],
    "sgdc__alpha": [2, 3, 4, 5],
    "sgdc__l1_ratio": ['scale', 'auto'],
    "sgdc__fit_intercept": [True, False],
    "sgdc__max_iter": [1000, 2000, 5000],
    "sgdc__tol": [1e-3, 1e-4, 1e-5],
    
    "dt__criterion": ['gini', 'entropy'],
    "dt__splitter": ['best', 'random'],
    "dt__max_depth": [None, 5, 10, 20],
    "dt__min_samples_split": [2, 5, 10],
    "dt__min_samples_leaf": [1, 2, 4],
    
    "rf__n_estimators": [10, 50, 100, 200],
    "rf__criterion": ['gini', 'entropy'],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__min_weight_fraction_leaf": [0, 0.1, 0.2],
    "rf__max_leaf_nodes": [None, 10, 20, 30],
    "rf__max_depth": [None, 5, 10],
    
    "knn__n_neighbours": [3, 5, 7, 9],
    "knn__weights": ['uniform', 'distance'],
    "knn__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
    
    "svc__C": [0.1, 1, 10, 100],
    "svc__kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "svc__degree": [2, 3, 4, 5],
    "svc__gamma": ['scale', 'auto'],
    
    "xgb__max_depth": [3, 5, 7, 9],
    "xgb__learning_rate": [0.1, 0.2, 0.3],
    "xgb__n_estimators": [100, 200, 300],
    "xgb__gamma": [0, 0.5, 1],
    "xgb__subsample": [0.5, 0.8, 1.0],
    "xgb__colsample_bytree": [0.5, 0.8, 1.0],
    "xgb__reg_alpha": [0, 0.5, 1],
    
    "nb__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
}
param_grid = {}


# cross-validating dataset
scrs = []
def cross_val_train():
    for model in models:
        for train, test in kfold.split(x_train):
            grid = GridSearchCV(
                model,
                param_grid,
                cv=kfold,
                scoring='neg_log_loss', refit='neg_log_loss'
            )
            grid.fit(x_train[train], y_train[train])
            
            scr = grid.score(x_train[test], y_train[test])
            print("score : {:.2f}".format(np.mean(scr)))
            scrs.append(scr)
            
            grid = GridSearchCV(
                model,
                param_grid,
                cv=kfold,
                scoring='roc_auc', refit='roc_auc'
            )
            grid.fit(x_train[train], y_train[train])
            
            y_pred = grid.predict(x_train[test])
            acc = metrics.accuracy_score(y_train[test], y_pred)
            print("accuracy : {:.2f} +/- {:.2f}".format(np.mean(acc), np.std(acc)))
    
            log = metrics.log_loss(y_train[test], y_pred)
            print("logloss : {:.2f}".format(np.mean(log)))
            
            print("")
            mtrx = metrics.confusion_matrix(y_train[test], y_pred)
            print("confusion matrix")
            print(mtrx)
            print("")
            print("")


print("")
cross_val_train()
print(sum(scrs) / len(scrs))
print("")

# --------------------------------------------------

    


# --------------------------------------------------
# hyperparameter tuning
param_grid = {
    "n_estimators": [10, 50, 100, 200],
    "criterion": ['gini', 'entropy'],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "min_weight_fraction_leaf": [0, 0.1, 0.2],
    "max_leaf_nodes": [None, 10, 20, 30],
    "max_depth": [None, 5, 10],
}
param_grid = {}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=kfold,
    scoring='roc_auc', refit='roc_auc'
)
grid.fit(x_train, y_train)
print(grid.best_params_)



import pickle
pickle.dump(grid, open('pickle/grid.pkl', 'wb'))
# --------------------------------------------------
