import pandas as pd
from sklearn.utils import shuffle
import random
from river import datasets
from classes import *
from IDA_frank_main_balance import *
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
from IDA_frank_main_balance import *

from river import datasets
import random
from sklearn.utils import shuffle

dataset = datasets.Phishing()
data = list(dataset)
data = shuffle(data, random_state=22)
# Definire le proporzioni
start_ratio_frank = 0.3
iterative_ratio = 0.4
test_ratio = 0.3
# Calcolare le dimensioni
n = len(data)
start_size = int(n * start_ratio_frank)
iterative_size = int(n*iterative_ratio)
test_size = int(n*test_ratio)

# Suddividere in sottoinsiemi
start_data = data[:start_size]
iterative_data = data[start_size:start_size+iterative_size]
test_data = data[iterative_size+test_size:]



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
df_test = pd.DataFrame(X_test)

cats = ["age_of_domain","ip_in_url"]
protected = df.columns.to_list()








rule_att = 'empty_server_form_handler'
rule_value = 0
df_train = pd.DataFrame(X_frank_train)
df_train[target] = Y_frank_train





    




rethink_value = 0.8 #Individual Fairness
fairness_value = 0.25 #top records to re-label (Group Fairness)
name_1 = dataset.filename.split('.')[0]



sgt= tree.SGTClassifier(
    feature_quantizer=tree.splitter.DynamicQuantizer(
       # n_bins=32, warm_start=10
    ),grace_period=10,nominal_attributes=cats
    )

ada = ensemble.AdaBoostClassifier(model=(sgt),n_models=5,seed=42 )
bagging = ensemble.BaggingClassifier(model=(sgt),n_models=5,seed=42 )
adwin_b = ensemble.ADWINBoostingClassifier(model  = (sgt),n_models=5,seed=42 )
models = [sgt,ada,bagging,adwin_b]
X = df.loc[:, df.columns != target]
X = list(X.to_dict(orient='index').values())
Y = df[target].values.tolist()

user_model_1 = GroundTruther(believe_level=0.8, rethink_level=rethink_value, fairness=fairness_value, expertise=0.8)
user_model_2 = GroundTruther(believe_level=0.5, rethink_level=rethink_value, fairness=fairness_value, expertise=0.8)
user_model_3 = GroundTruther(believe_level=0.2, rethink_level=rethink_value, fairness=fairness_value, expertise=0.8)
users = [user_model_2,user_model_3]

for user in range(0,len(users)):
    ada = ensemble.AdaBoostClassifier(model=(sgt),n_models=5,seed=42 ) 
    name = str(user)+"user"+name_1
    f = Frank_balance(name_1,ada,users[user], df, target, protected, cats, rule_att, rule_value, RULE, PAST, SKEPT, GROUP, X_test, Y_test, EVA, N_BINS, N_VAR, MAX,X_frank_train,Y_frank_train,user)
    f.train(X_frank_train, Y_frank_train)
    results, eva,accuracy_score,f1_score,equality = f.start()
