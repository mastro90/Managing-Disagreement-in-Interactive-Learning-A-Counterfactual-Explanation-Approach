import pandas as pd
from sklearn.utils import shuffle
import random
from river import datasets
from classes import *
from IDA_frank_main_unbalance import *
EVA = True

import itertools
import numpy as np
from itertools import permutations 
import json

from classes import *


EVA = False
from river import ensemble
from river import preprocessing
from river import linear_model
from sklearn.utils import shuffle
RULE = False
PAST = False
SKEPT = True
GROUP = False

N_BINS = 10
N_VAR = 3
MAX = 5

from river import datasets
import random
from sklearn.utils import shuffle

dataset = datasets.CreditCard()
data = list(dataset)
data = shuffle(data, random_state=2)
# Definire le proporzioni
start_ratio_frank = 0.2
iterative_ratio = 0.4
test_ratio = 0.4
# Calcolare le dimensioni
n = len(data)
start_size = int(n * start_ratio_frank)
iterative_size = int(n*iterative_ratio)
test_size = int(n*test_ratio)

# Suddividere in sottoinsiemi
start_data = data[:start_size]
iterative_data = data[start_size:start_size+iterative_size]
test_data = data[iterative_size+test_size:]


# Separare feature (X) e target (y)
X_frank_train, Y_frank_train = zip(*start_data)
Y_frank_train = [bool(y) for y in Y_frank_train]


X_iterative_train,Y_iterative_train = zip(*iterative_data)
Y_iterative_train = [bool(y) for y in Y_iterative_train]
X_test, Y_test = zip(*test_data)
Y_test = [bool(y) for y in Y_test]

target = 'target'
#Creazione dataset per Frank
df = pd.DataFrame(X_iterative_train)
df[target] = Y_iterative_train

cats = []
protected = df.columns.to_list()

rule_att = 'Time'
rule_value = 0
df_train = pd.DataFrame(X_frank_train)
df_train[target] = Y_frank_train
df_test = pd.DataFrame(X_test)
df_test[target] = Y_test
# Estrazione   gruppi
true_df = df[df['target'] == True]
false_df = df[df['target'] == False]

# Scelta  proporzione 
n_true = min(len(true_df), 100)  # 100 positives
n_false = n_true * 9             # 900 negativi

# Campionamento casuale
sampled_true = true_df.sample(n=n_true, random_state=42)
sampled_false = false_df.sample(n=n_false, random_state=42)

# Unione 
balanced_iterative = pd.concat([sampled_true, sampled_false]).sample(frac=1, random_state=42)
print(df_train[target].value_counts())
# Verifica della distribuzione
print(balanced_iterative['target'].value_counts())

# Selezionamento positives esclusi dal sottoinsieme di training
positives_iterative = df[df['target'] == True]
positives_used = sampled_true  # quelli used nel training iterative ridotto

# positives non used
positives_non_used = positives_iterative[~positives_iterative.index.isin(positives_used.index)]


df_test_clean = df_test[~df_test.index.isin(positives_non_used.index)]

# Aggiungi i positives scartati al test set
df_test_esteso = pd.concat([df_test_clean, positives_non_used]).sample(frac=1, random_state=42)

# Verifica nuova distribuzione
print(df_test_esteso['target'].value_counts())
X_frank_train_1 = balanced_iterative.drop(columns=['target']).to_dict(orient='records')
Y_frank_train_1 = balanced_iterative['target'].tolist()

X_test = df_test_esteso.drop(columns=['target']).to_dict(orient='records')
Y_test = df_test_esteso['target'].tolist()


X_test_tuples = set(tuple(sorted(d.items())) for d in X_frank_train_1)


X_test_1 = []
Y_test_1 = []

for x, y in zip(X_test, Y_test):
    t = tuple(sorted(x.items()))
    if t not in X_test_tuples:
        X_test_1.append(dict(t)) 
        Y_test_1.append(y)
       

count_true = 0
count_false = 0
for y in Y_test_1:
    if y==True:
        count_true+=1
    else:
        count_false+=1
print('True',count_true)
print('False',count_false)            












rethink_value = 0.8 #Individual Fairness
fairness_value = 0.25 #top records to re-label (Group Fairness)
name_1 = dataset.filename.split('.')[0]






sgt= tree.SGTClassifier(
    feature_quantizer=tree.splitter.DynamicQuantizer(
       # n_bins=32, warm_start=10
    ),grace_period=10
    )


ada = ensemble.AdaBoostClassifier(model=(sgt),n_models=3,seed=42 )
bagging = ensemble.BaggingClassifier(model=(sgt),n_models=3,seed=42 )
adwin_b = ensemble.ADWINBaggingClassifier(model  = (sgt),n_models=3,seed=42 )
models = [ada,bagging,adwin_b]
X = balanced_iterative.loc[:, df.columns != target]
X = list(X.to_dict(orient='index').values())
Y = balanced_iterative[target].values.tolist()

for x,y in zip(X_frank_train_1,Y_frank_train_1):
    bagging.learn_one(x,y)
for x,y in zip(X,Y):
    y_pred = bagging.predict_one(x)
    bagging.learn_one(x,y_pred)
acc,f1,frank_cm =calculate_metrics(X_test_1,Y_test_1,bagging)      
print(bagging,'--->','acc:',acc,'f1:','--->',f1)        
    
user_model_1 = GroundTruther(believe_level=0.8, rethink_level=rethink_value, fairness=fairness_value, expertise=0.8)
user_model_2 = GroundTruther(believe_level=0.5, rethink_level=rethink_value, fairness=fairness_value, expertise=0.8)
user_model_3 = GroundTruther(believe_level=0.8, rethink_level=rethink_value, fairness=fairness_value, expertise=0.5)
user_model_4 = GroundTruther(believe_level=0.5, rethink_level=rethink_value, fairness=fairness_value, expertise=0.5)
users = [user_model_1,user_model_2,user_model_3,user_model_4]


for user in range(0,len(users)):
    bagging = ensemble.BaggingClassifier(model=(sgt),n_models=3,seed=42 )
    name = str(user)+"user"+name_1
    f = Frank_unbalance(name_1, bagging,users[user],balanced_iterative, target, protected, cats, rule_att, rule_value, RULE, PAST, SKEPT, GROUP, X_test_1, Y_test_1, EVA, N_BINS, N_VAR, MAX,X_frank_train_1,Y_frank_train_1,user)
    f.train(X_frank_train_1, Y_frank_train_1)
    results, eva,accuracy_score,f1_score,equality = f.start()
