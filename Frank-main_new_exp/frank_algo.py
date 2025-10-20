from river import rules, tree, datasets, drift, metrics, evaluate
from IPython import display

import random
import functools
from itertools import combinations, product


import numpy as np
import time

import pandas as pd
from sklearn.compose import ColumnTransformer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,f1_score

import fatf.utils.data.datasets as fatf_datasets

import fatf.fairness.data.measures as fatf_dfm

import fatf.utils.data.tools as fatf_data_tools

from tqdm import tqdm
from matplotlib import pyplot as plt

from classes import *
import dice_ml
from dice_ml import Dice
import pandas as pd
import sys
from growingspheres import counterfactuals as cf

import copy
import pprint
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras.callbacks import EarlyStopping
#from datasets import load_dataset
from sklearn.metrics import classification_report

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pdb
from coipee.src.coipee import Coipee
from scipy.spatial.distance import cdist
#from ceml.sklearn.models import generate_counterfactual
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
import tensorflow as tf
from alibi.models.tensorflow import HeAE
from alibi.models.tensorflow import Actor, Critic
from alibi.models.tensorflow import ADULTEncoder, ADULTDecoder
from alibi.explainers.cfrl_base import Callback
from alibi.explainers.backends.cfrl_tabular import get_he_preprocessor, get_statistics, \
    get_conditional_vector, apply_category_mapping
from alibi.models.tensorflow import HeAE
from alibi.models.tensorflow import Actor, Critic
from alibi.models.tensorflow import ADULTEncoder, ADULTDecoder
from alibi.explainers import CounterfactualRLTabular, CounterfactualRL
from tensorflow import keras
from alibi.explainers import CounterfactualProto
from sklearn.preprocessing import StandardScaler


from alibi.explainers.cfrl_base import Callback
from alibi.explainers.backends.cfrl_tabular import get_he_preprocessor, get_statistics, \
    get_conditional_vector, apply_category_mapping
from river import imblearn
from river import preprocessing
from river import optim

from xailib.data_loaders.dataframe_loader import prepare_dataframe

from xailib.explainers.lime_explainer import LimeXAITabularExplainer
from xailib.explainers.lore_explainer import LoreTabularExplainer
from xailib.explainers.shap_explainer_tab import ShapXAITabularExplainer

from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
import signal
import time

def get_index (attribute, attr_list):
    for j in range(len(attr_list)):
        if attr_list[j] == attribute:
            return j
        
def percentage(value, all_records):
    return round((value * all_records) / 100)

def ideal_record_test(rec, rule_att, rule_value):
    if rec[rule_att] > rule_value:
        return True
    else:
        return None
    
def get_value_swap_records(x, processed, protected, attr_list):
    
    protected_inx = []
    for att in protected:
        protected_inx.append(get_index(att, attr_list))
                
    current = list(x.values())
    vs_records = []
    check = True
    vs_decision = None
    
    for record in list(processed.keys()):
        check = True
        for i in range(len(record)):
            if i not in protected_inx:
                if record[i] != current[i]:
                    check = False
            else:
                if record[i] == current[i]:
                    check = False
        if check:
            vs_records.append(record)
            vs_decision = processed[record]['decision']
    
    return vs_records, vs_decision

def get_fairness(model, protected, processed, protected_values):
    PP, PN, DP, DN = [], [], [], []
    PP_c, PN_c, DP_c, DN_c = 0, 0, 0, 0

    for rec in list(processed.keys()):
        og_rec = processed[rec]['dict_form']
        proba = model.predict_proba_one(og_rec)[True]
        if processed[rec]['decision'] == True:
            if processed[rec]['dict_form'][protected[0]] == protected_values[0]:
                PP_c = PP_c + 1
                if processed[rec]['vs'] is None and processed[rec]['ideal'] is None:
                    PP.append(((proba, rec)))
            else:
                DP_c = DP_c + 1
                if processed[rec]['vs'] is None and processed[rec]['ideal'] is None:
                    DP.append(((proba, rec)))
        else:
            if processed[rec]['dict_form'][protected[0]] == protected_values[0]:
                PN_c = PN_c + 1
                if processed[rec]['vs'] is None and processed[rec]['ideal'] is None:
                    PN.append(((proba, rec)))
            else:
                DN_c = DN_c + 1
                if processed[rec]['vs'] is None and processed[rec]['ideal'] is None:
                    DN.append(((proba, rec)))
                  
    try:
        fairness = (PP_c) / ((PP_c)+(PN_c)) - (DP_c) / ((DP_c)+(DN_c))
    except:
        fairness = 0
    
    if fairness != 0:
        fair_number = round(((DP_c)+(DN_c)) * ((PP_c)+(DP_c)) / ((PP_c)+(PN_c)+(DP_c)+(DN_c)))
        
    if fairness < 0:
        DN = PN
        PP = DP

    DN = [e for e in DN if e[0] > 0.5]
    PP = [e for e in PP if e[0] < 0.5]
    
    DN = sorted(DN, reverse=True)
    PP = sorted(PP)
    
    return DN, PP, fairness

def evaluation_human (processed, protected, Y, attr_list):
    DN, DP, PN, PP = 0, 0, 0, 0
    Y_final = []

    for r in processed.keys():
        
        record = processed[r]['dict_form']
        sa = record[protected[0]]
        decision = processed[r]['decision']
        
        Y_final.append(decision) #for accuracy

        if decision == 0:
            if sa == 0:
                PN = PN + 1
            else:
                DN = DN + 1
        else:
            if sa == 0:
                PP = PP + 1
            else:
                DP = DP + 1

    try:
        human_fairness = (PP) / ((PP)+(PN)) - (DP) / ((DP)+(DN))
    except:
        human_fairness = 0
        
    human_acc = accuracy_score(Y_final, Y[:len(Y_final)])
    
    processed_df = pd.DataFrame.from_dict(list(processed.keys()))
    processed_df.columns = attr_list[:-1]
    data_fairness_matrix = fatf_dfm.systemic_bias(np.array(list(processed_df.to_records(index=False))), np.array(Y_final), protected)
    is_data_unfair = fatf_dfm.systemic_bias_check(data_fairness_matrix)
    unfair_pairs_tuple = np.where(data_fairness_matrix)
    unfair_pairs = []
    for i, j in zip(*unfair_pairs_tuple):
        pair_a, pair_b = (i, j), (j, i)
        if pair_a not in unfair_pairs and pair_b not in unfair_pairs:
            unfair_pairs.append(pair_a)
    if is_data_unfair:
        unfair_n = len(unfair_pairs)
    else:
        unfair_n = 0
        
    return human_fairness, human_acc, unfair_n


def evaluation_frank (X_test, Y_test, model, protected):
    frank_preds = []

    PP, DP, PN, DN = 0, 0, 0, 0

    for x_t, y_t in zip(X_test, Y_test):

        test_pred = model.predict_one(x_t)
        #print('test pred : ',test_pred)
        frank_preds.append(test_pred)
        

        if test_pred == True:
            if x_t[protected[0]] == 0: #0 Male, 1 Female in our tests
                PP = PP + 1
            else:
                DP = DP + 1
        else:
            if x_t[protected[0]] == 0:
                PN = PN + 1
            else:
                DN = DN + 1

    try:
        frank_fairness = (PP) / ((PP)+(PN)) - (DP) / ((DP)+(DN))
    except:
        frank_fairness = 0
    #print('frank_preds,Y_test:',frank_preds, Y_test)
    frank_acc = accuracy_score(frank_preds, Y_test)
    frank_f1 = f1_score(Y_test, frank_preds, average='macro')
    #print('accuracy',frank_acc)
    from sklearn.metrics import confusion_matrix
    frank_cm = confusion_matrix(Y_test, frank_preds)
    #print('Confusion_matrix : ',confusion_matrix(Y_test, frank_preds))
    
    return frank_fairness, frank_acc,frank_f1,frank_cm

def get_examples(processed, x, model, attr_list, cats, N_BINS, N_VAR, MAX):
    processed_df = pd.DataFrame.from_dict(list(processed.keys()))
    processed_df.columns = attr_list[:-1]

    binned_X = processed_df.copy()
    feats = dict()
    for f in processed_df.columns:
        if f in cats:
            feats[f] = processed_df[f].unique()
        else:
            if len(processed_df[f].unique()) <= N_BINS:
                feats[f] = processed_df[f].unique()
            else:
                binned_X['bins'] = pd.cut(processed_df[f], N_BINS)
                binned_X['median'] = binned_X.groupby('bins')[f].transform('median')
                feats[f] = binned_X['median'].unique()


    all_combinations = []
    for i in range(N_VAR):
        combination = []
        for feat_comb in combinations(feats.keys(), i+1):
            combination.append(feat_comb)
        all_combinations.append(combination)


    ok_feats_against = []
    all_cf_against = []
    ok_feats_pro = []
    all_cf_pro = []
    
    for combination in all_combinations:
        for feat_set in combination:
            if len([f for f in feat_set if f in ok_feats_against]) == 0 and len(all_cf_against) < MAX:
                #print (feat_set)
                list_of_values = []
                for f in feat_set:
                    list_of_values.append(feats[f])
                cf_x = x.copy()

                for val_comb in product(*list_of_values):
                    if len([f for f in feat_set if f in ok_feats_against]) == 0 and len(all_cf_against) < MAX:
                        for val, f in zip(val_comb, feat_set):
                            #idx = list(feats.keys()).index(f)
                            #print(f, idx, "--->", val)
                            cf_x[f] = val
                            #cf_x_model.at[0,f]=val
                        #print(np.array(cf_x))
                        #print("")
                        if model.predict_one(cf_x) == model.predict_one(x) and list(cf_x.values()) != list(x.values()):
                        # == as they are counterfactual AGAINST THE USER'S DECISION
                        # if we want against the machine =!
                        # second condition to avoid having the same record
                            all_cf_against.append(cf_x)
                            for f in feat_set:
                                ok_feats_against.append(f)
            if len([f for f in feat_set if f in ok_feats_pro]) == 0 and len(all_cf_pro) < MAX:
                #print (feat_set)
                list_of_values = []
                for f in feat_set:
                    list_of_values.append(feats[f])
                cf_x = x.copy()

                for val_comb in product(*list_of_values):
                    if len([f for f in feat_set if f in ok_feats_pro]) == 0 and len(all_cf_pro) < MAX:
                        for val, f in zip(val_comb, feat_set): #### controllare se cambiano 2+ feats 
                            #idx = list(feats.keys()).index(f)
                            #print(f, idx, "--->", val)
                            cf_x[f] = val
                            #cf_x_model.at[0,f]=val
                        #print(np.array(cf_x))
                        #print("")
                        if model.predict_one(cf_x) != model.predict_one(x) and list(cf_x.values()) != list(x.values()):
                        # != as they are counterfactual IN FAVOR OF THE USER'S DECISION
                        # if we want against the machine =!
                        # second condition to avoid having the same record
                            all_cf_pro.append(cf_x)
                            for f in feat_set:
                                ok_feats_pro.append(f)


    #print("These records are similar, and should be labelled:", model.predict_one(cf_x))
    #print(attr_list)
    #for e in all_cf:
        #print (list(e.values()))
    return all_cf_pro, all_cf_against



def timeout_handler(signum, frame):
    raise TimeoutError("Tempo scaduto!")








def get_counterfactuals_FatForensic(X_frank_train ,Y_frank_train,processed,x, model, target,cats):
    """
    Genera spiegazioni controfattuali per un record  `x` utilizzando Fat-Forensics.
    """
    df_to_explain = pd.DataFrame([x])
    df_train = pd.DataFrame(X_frank_train)
    df_train[target] = Y_frank_train
    processed_c = processed.copy()
    rows = []
    df_log = pd.DataFrame()
    if len(processed_c)>0:#Trasformo lo storico in un DataFrame se ci sono degli esempi visti
        for entry in processed_c.values():
            row = entry['dict_form'].copy()
            row[target] = entry['decision']
            rows.append(row)
        df_proc = pd.DataFrame(rows)
        df_log =  pd.concat([df_train,df_proc], ignore_index=True)#Concateno lo storico con il dataframe di train
    else:# Altriemnti uso solamente il dataframe usato per il train
        df_log = df_train

    feature_names = list(df_log.columns)
    query_instance = df_to_explain.values
    river_model_wrapper = RiverModelWrapper(model,target,feature_names)   
    n_nb = 100
    df_log = df_log.drop([target], axis = 1)
    max_iter = 100 
    start_time = time.time()
    
    #Per velocizzare la computazione , prendo gli esempi più simili alla query_instance
    #nbrs = NearestNeighbors(n_neighbors=n_nb).fit(df_log.to_numpy())
    #_, indices = nbrs.kneighbors([query_instance.flatten()])
    #reduced_df = df_log.iloc[indices[0]]
    
   
            
       
    
    #Salvo gli indici numerici delle varie tipologie di feature
    indices_cat = []
    indices_cat_c = []
    if  cats !=[]:
        indices_cat = [df_log.columns.get_loc(col) for col in cats ]
    else :
          indices_cat = None  
    cont_feature = [f for f in feature_names if f not in cats and f!=target]
    indices_cont = [df_log.columns.get_loc(col) for col in cont_feature ]
    
    
    
    

    import fatf.transparency.predictions.counterfactuals as fatf_cf
    #Array utili per salvare le varie info di un controfattuale
    dp_1_cfs  = np.array([])
    dp_1_cfs_distances= np.array([])
    dp_1_cfs_predictions= np.array([])
    if cats!=[]:
        indices_cat_c = indices_cat.copy()# Copia degli indici  delle feature catgoriche
    else:
        indices_cat_c = indices_cont.copy()# Copia degli indici  delle feature continue    
   
    max_len = 1#Numero di feature alterate per comporre un controfattuale
    step_size = 0.5
   
    iter = 0

    while len(dp_1_cfs)==0 and iter<max_iter :
        try:    
            cf_explainer = fatf_cf.CounterfactualExplainer(
            model= river_model_wrapper,
            dataset=df_log.to_numpy(),
            categorical_indices=indices_cat ,
            numerical_indices=indices_cont,
            counterfactual_feature_indices =indices_cat_c ,#Inizio utilizzando solamente le feature categoriche
            max_counterfactual_length = max_len,
            default_numerical_step_size=step_size)
   
            dp_1_cf_tuple = cf_explainer.explain_instance( query_instance.flatten())
            dp_1_cfs, dp_1_cfs_distances, dp_1_cfs_predictions = dp_1_cf_tuple
            
       
            print('Lunghezza',max_len)
            max_len += 1#Incremento del numero di feature
            #Se sono state utilizzate tutte le feature e non è stato prodotto alcun controfattuale , utilizzo le feature continue
            if len(feature_names) == max_len and indices_cat_c != indices_cont:
                print('max_superato')      
                indices_cat_c = indices_cont+indices_cat_c
                max_len =1
                step_size += 0.5
                n_nb = 100
                #nbrs = NearestNeighbors(n_neighbors=n_nb).fit(df_log.to_numpy())
                #_, indices = nbrs.kneighbors([query_instance.flatten()])
                #reduced_df = df_log.iloc[indices[0]]
            #Se sono state utilizzate tutte le feature , non è stato possibile trovare alcun controfattuale e sto utilizzando le feature continue , cambio il passo    
            elif len(feature_names) == max_len and indices_cat_c == indices_cont:
                print('aumento_step_size')
                step_size += 0.5
                max_len = 1
            iter+=1 
            if iter==max_iter:
                return None   

            
            
        #Se una particolare feature ha lo stesso value in tutte le righe del dataframe , cambio il numero di vicini    
        except ValueError as e:
            print(e)
            #indices_cat_c = indices_cont
            #step_size+=0.05
            # n_nb+=10
            # nbrs = NearestNeighbors(n_neighbors=n_nb).fit(df_log.to_numpy())
            # _, indices = nbrs.kneighbors([query_instance.flatten()])
            # reduced_df = df_log.iloc[indices[0]]    
            #if max_len!=1:#Rinizializzo il numero di feature
               # max_len = 3
            
               

            
       
       
       
       
       
       
       
       

            

        

    
    dp_1_cfs, dp_1_cfs_distances, dp_1_cfs_predictions = dp_1_cf_tuple
    
    print('Controfattuali trovati')
    print('Controfattuali : ',dp_1_cfs)
    elapsed_time = time.time() - start_time
    sparsities = np.sum(dp_1_cfs != query_instance, axis=1)
    print('Sparsità:',sparsities)
   
    print('Distanze : ',dp_1_cfs_distances)
    print('Predizioni : ', dp_1_cfs_predictions)
    print(f"Tempo impiegato: {elapsed_time:.2f} secondi")
    cfs = [dict(zip(feature_names, riga)) for riga in dp_1_cfs]
    print(cfs)
    return dp_1_cfs,dp_1_cfs_distances,dp_1_cfs_predictions,cfs,elapsed_time,sparsities
   

    

   
    




def get_counterfactuals_DICE(df_log, x, model, cats, target):
    """
    Genera spiegazioni controfattuali per un record  `x` utilizzando DICE.
    """
    query_instance = pd.DataFrame([x])
    start_time = time.time()

    df_log[target] = [bool(y) for y in df_log[target]]
    feature_names = [f for f in df_log.columns if f != target]
   

    river_model_wrapper = RiverModelWrapper(model, target)

    if cats != []:
        dice_data = dice_ml.Data(
            dataframe=df_log,
            categorical_features=cats,
            continuous_features=[f for f in df_log.columns if f not in cats and f != target],
            outcome_name=target
        )
        dice_model = dice_ml.Model(model=river_model_wrapper, backend="sklearn")
        dice_exp = dice_ml.Dice(dice_data, dice_model, method="random")
    else:
        dice_data = dice_ml.Data(
            dataframe=df_log,
            continuous_features=[f for f in df_log.columns if f != target],
            outcome_name=target
        )
        dice_model = dice_ml.Model(model=river_model_wrapper, backend="sklearn")
        dice_exp = dice_ml.Dice(dice_data, dice_model, method="kdtree")

    # Tentativi progressivi aumentando le feature da variare
    for i in range(1, len(feature_names)+1):
        features_to_vary = feature_names[:i]
        #print(f"Tentativo con le prime {i} feature: {features_to_vary}")
        try:
            counterfactuals = dice_exp.generate_counterfactuals(
                query_instance,
                total_CFs=1,
                desired_class="opposite",
                verbose=True,
                features_to_vary=features_to_vary
            )
            print('Controfattuali trovati')
            elapsed_time = time.time() - start_time

            cfs = counterfactuals.cf_examples_list[0].final_cfs_df.drop([target], axis=1).to_dict(orient="records")
            cf_x = counterfactuals.cf_examples_list[0].final_cfs_df.drop([target], axis=1).values[0]

            sparsity = np.sum(np.array(list(x.values())) != cf_x)
            return cfs[0], elapsed_time, sparsity

        except Exception as e:
            #print(f"Tentativo fallito con {i} feature: {e}")
            continue

    # Fallback: se nessun controfattuale è stato trovato, usa l'esempio più simile con classe opposta
    #print("Controfattuali non trovati. Passo al fallback con l'esempio più simile.")
    X = df_log.loc[:, df_log.columns != target]
    X = list(X.to_dict(orient='index').values())

    min_d = float('inf')
    x_ = None
    for x_h in X:
        if model.predict_one(x) != model.predict_one(x_h):
            d = cdist(query_instance, pd.DataFrame([x_h]), metric='euclidean').flatten()
            if d[0] < min_d:
                min_d = d[0]
                x_ = x_h

    elapsed_time = time.time() - start_time
    sparsity = np.sum(np.array(list(x.values())) != np.array(list(x_.values())))
    return x_, elapsed_time, sparsity













def counts (processed,Y_frank_train):
    """
    Conta gli esempi con il quale il modello viene addestrato
 
    """

    decision_final = []
    if len(processed)>0:
        for k , result in processed.items():
            decision_final.append(result['decision'])

    i_true = 0
    i_false = 0
    i_true_y = 0
    i_false_y = 0
   
    for i in Y_frank_train :
        if i == False:
            i_false_y+=1
        elif i == True:
            i_true_y +=1

    if len(decision_final)>0: 
        for i in decision_final:
            if i == False:
                i_false+=1
            elif i == True:
                i_true+=1    

    print('False_decision:',(i_false+i_false_y))
    print('True_decision:',(i_true+i_true_y))
    





    






















def calculate_metrics(X_test, Y_test, model):
    frank_preds = [model.predict_one(x_t) for x_t in X_test]
    #print(frank_preds)
    frank_acc = accuracy_score(Y_test, frank_preds)
    #print(Y_test)
    frank_f1 = f1_score(Y_test, frank_preds, average='macro')
   
    from sklearn.metrics import confusion_matrix
    frank_cm = confusion_matrix(Y_test, frank_preds)
    

    return frank_acc,frank_f1,frank_cm













    
    
    
    
    
    

    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    


def make_ae(scaler,ENCODING_DIM,X_train,Y_train,BATCH_SIZE,EPOCHS):
    len_input_output = X_train.shape[-1]

    encoder = keras.Sequential()
    encoder.add(Dense(units=ENCODING_DIM*2, activation="relu", input_shape=(len_input_output, )))
    encoder.add(Dense(units=ENCODING_DIM, activation="relu"))

    decoder = keras.Sequential()
    decoder.add(Dense(units=ENCODING_DIM*2, activation="relu", input_shape=(ENCODING_DIM, )))
    decoder.add(Dense(units=len_input_output, activation="linear"))

    ae = AE(encoder=encoder, decoder=decoder)
    #dataset = tf.data.Dataset.from_tensor_slices((scaler.transform(X_train), Y_train)).batch(BATCH_SIZE)

    ae.compile(optimizer='adam', loss='mean_squared_error')
    ae.fit(
       scaler.fit_transform(X_train),
       scaler.fit_transform(X_train),
        #X_train,
        #X_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=False)
   
    return ae,scaler
   
   

   
   

def get_examples_5(ae,cats,x,df_log,X_frank_train,Y_frank_train,target,model):
    df_train = pd.DataFrame(X_frank_train)
    df_train[target] = Y_frank_train
    #processed_c = processed.copy()
    #print('Processed:',processed_c)
    rows = []
    query_instance = pd.DataFrame([x])
    #percentage_dict,df_log = get_percentage_and_df(X_frank_train,Y_frank_train,processed_c,target)
    feature_names = df_log.columns.tolist()
    print(df_log.iloc[-1])

    print(x)

    
   

    river_model_wrapper = RiverModelWrapper(model,target,feature_names)
    predictor = lambda x:  river_model_wrapper.predict_proba(x)
    Y = df_log[target].values
    X = df_log.drop(target,axis = 1)
    feature_names = X.columns.tolist()   
    X = X.to_numpy()
    HIDDEN_DIM = X.shape[0] 
    EPOCHS = 80 # epochs to train the autoencoder
    LATENT_DIM = 10 # define latent dimension
    COEFF_SPARSITY = 0.8              # sparisty coefficient
    COEFF_CONSISTENCY = 1.5            # consisteny coefficient
    TRAIN_STEPS = 1000              # number of training steps -> consider increasing the number of steps
    BATCH_SIZE = X.shape[0] -10                  # batch size
    start_time = time.time()
    elpased_time = 0
    
    if cats != []:
        category_map =  {}
        for i in range(len(feature_names)):
            if feature_names[i]  in cats:
                    category_map[i] = df_log[feature_names[i]].unique().tolist()
       

        # Separate columns in numerical and categorical.
        categorical_names = [feature_names[i] for i in category_map.keys()]
        categorical_ids = list(category_map.keys())
        numerical_names = [name for i, name in enumerate(feature_names) if i not in category_map.keys()]
        numerical_ids = [i for i in range(len(feature_names)) if i not in category_map.keys()]
        heae_preprocessor, heae_inv_preprocessor = get_he_preprocessor(X=X,
                                                               feature_names=feature_names,
                                                               category_map=category_map)
        trainset_input = heae_preprocessor(X).astype(np.float32)
        trainset_outputs = {
        "output_1": trainset_input[:, :len(numerical_ids)]
        }
        for i, cat_id in enumerate(categorical_ids):
            trainset_outputs.update({
            f"output_{i+2}": X[:, cat_id]
            })
        
        trainset = tf.data.Dataset.from_tensor_slices((trainset_input, trainset_outputs))
        trainset = trainset.shuffle(1024).batch(HIDDEN_DIM  , drop_remainder=True)
        # Define output dimensions.
        OUTPUT_DIMS = [len(numerical_ids)]
        OUTPUT_DIMS += [len(category_map[cat_id]) for cat_id in categorical_ids]

        # Define the heterogeneous auto-encoder.
        heae = HeAE(encoder=ADULTEncoder(hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM),
            decoder=ADULTDecoder(hidden_dim=HIDDEN_DIM, output_dims=OUTPUT_DIMS))
         # Define loss functions.
        he_loss = [keras.losses.MeanSquaredError()]
        he_loss_weights = [1.]

        # Add categorical losses.
        for i in range(len(categorical_names)):
            he_loss.append(keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            he_loss_weights.append(1./len(categorical_names))

        # Define metrics.
        metrics = {}
        for i, cat_name in enumerate(categorical_names):
            metrics.update({f"output_{i+2}": keras.metrics.SparseCategoricalAccuracy()})

        # Compile model.
        heae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
             loss=he_loss,
             loss_weights=he_loss_weights,
             metrics=metrics)

        heae.fit(trainset, epochs=EPOCHS)
     # Define constants
        desired = int(not(model.predict_one(x)))
        instance_class_prediction = model.predict_one(x)
        counter_factual_pred = instance_class_prediction
        
        while  desired!= counter_factual_pred and TRAIN_STEPS<=10000:
            explainer = CounterfactualRLTabular(predictor=predictor,
                                    encoder=heae.encoder,
                                    decoder=heae.decoder,
                                    latent_dim=LATENT_DIM,
                                    encoder_preprocessor=heae_preprocessor,
                                    decoder_inv_preprocessor=heae_inv_preprocessor,
                                    coeff_sparsity=COEFF_SPARSITY,
                                    coeff_consistency=COEFF_CONSISTENCY,
                                    category_map=category_map,
                                    feature_names=feature_names,
                                    #ranges=ranges,
                                    #immutable_features=immutable_features,
                                    train_steps=TRAIN_STEPS,
                                    batch_size=BATCH_SIZE,
                                    backend="tensorflow")
        
            explainer = explainer.fit(X=X)
        #print(x)
            Y_t = np.array([int(not(model.predict_one(x)))])
            #print('Predizione originale:',Y_t)
            explanation = explainer.explain(query_instance.values, Y_t, [])
        # Concat labels to the counterfactual instances.
            cf = np.concatenate(
            [explanation.data['cf']['X'], explanation.data['cf']['class']],
            axis=1
            )
            print('cf_with class:',cf)
            cf = explanation.data['cf']['X']
            feature_dict = dict(zip(feature_names, cf.flatten()))
            counter_factual_pred = model.predict_one(feature_dict)
            if desired != counter_factual_pred:
                print('sono uguali')
                TRAIN_STEPS+=1000
                if COEFF_SPARSITY > 0.1:  # Limite minimo per evitare sparsity troppo bassa
                   COEFF_SPARSITY *= 0.9
                if BATCH_SIZE==X.shape[0]:
                    BATCH_SIZE-=10
                else:
                    BATCH_SIZE+=10 
            else:   
                elpased_time = time.time()-start_time
                def_sparsity = np.sum(query_instance.values != cf)
                print('Tempo : ',elpased_time)
                print('Sparsity : ',def_sparsity)
                return feature_dict,elpased_time,def_sparsity
        X = df_log.loc[:, df_log.columns != target]
        X = list(X.to_dict(orient='index').values())
        min_d = float('inf')
        x_ = None
        for x_h in X:
            print(x_h,":",model.predict_one(x_h))
            print(x,":",model.predict_one(x))
            if model.predict_one(x) != model.predict_one(x_h):      
                d = cdist(query_instance, pd.DataFrame([x_h]), metric='euclidean').flatten()
                if d[0] < min_d:
                    min_d = d[0]
                    x_ = x_h   
        if x_:
            cfs = x_
            elpased_time = time.time()-start_time
            def_sparsity = np.sum(x != cfs)
            return cfs, elpased_time,def_sparsity        


    else:
      print('Model:',model)
      desired = int(not(model.predict_one(x)))
      instance_class_prediction = model.predict_one(x)
      counter_factual_pred = instance_class_prediction
      sparsity = 0.1
      consistency = 0.1
      scaler = StandardScaler()
      scaler.fit(X)
      BATCH_SIZE = X.shape[0] -10#da cambiare
      from collections import Counter
      preds = [model.predict_one(row) for row in df_log.drop(columns=[target]).to_dict(orient="records")]
      print(Counter(preds))  
      river_model_wrapper = RiverModelWrapper(model,target,feature_names)
      predictor = lambda x:  river_model_wrapper.predict_proba(x)
      TRAIN_STEPS = 20000
      while desired!= counter_factual_pred and TRAIN_STEPS<=22000:
        #ENCODING_DIM = 10
       
        #percentage_dict = (df_log['target'].value_counts(normalize=True) * 100).round(2).to_dict()
        
        
        #ae = make_ae(scaler,ENCODING_DIM,X,Y,BATCH_SIZE,EPOCHS)
       
        

        cfrl_explainer = CounterfactualRL(
        predictor=predictor,               # The model to explain
        encoder=ae.encoder,                 # The encoder
        decoder=ae.decoder,                 # The decoder
        latent_dim=LATENT_DIM,              # The dimension of the autoencoder latent space
        coeff_sparsity=sparsity,                 # The coefficient of sparsity
        coeff_consistency=consistency,              # The coefficient of consistency
        train_steps=TRAIN_STEPS,            # The number of training steps
        batch_size=BATCH_SIZE               # The batch size
        )        
       
       #print(X)
        cfrl_explainer.fit(X)#=scaler.transform(X))
        print(x)
        

        Y_t = np.array([int(not(model.predict_one(x)))])
        #print('Y_T:',Y_t)
        explanation = cfrl_explainer.explain(X = query_instance.values, Y_t = Y_t,batch_size=BATCH_SIZE)
       
        # Concat labels to the counterfactual instances.
        print("Instance class prediction:", model.predict_one(x))
        cf = explanation.data['cf']['X']
        def_sparsity = np.sum(query_instance.values != cf)
        #print(explanation.data)
        #cf = explanation.data['cf']['X']
        feature_dict = dict(zip(feature_names, cf.flatten()))
        counter_factual_pred = model.predict_one(feature_dict)
        #print(feature_dict)
        print(model.predict_one(feature_dict))
        print('Sparsity: ',def_sparsity)
        if desired != counter_factual_pred:
            print('sono uguali')
            TRAIN_STEPS+=1000
            sparsity *= 0.75
            consistency *= 0.9
            if BATCH_SIZE==X.shape[0]:
                BATCH_SIZE-=10
            else:
        
                BATCH_SIZE+=10
                
        else:
            elpased_time = time.time()-start_time
            print('Tempo : ', elpased_time)    
            return feature_dict,elpased_time,def_sparsity

    X = df_log.loc[:, df_log.columns != target]
    X = list(X.to_dict(orient='index').values())
    min_d = float('inf')
    x_ = None
    for x_h in X:
            print(x_h,":",model.predict_one(x_h))
            print(x,":",model.predict_one(x))
            if model.predict_one(x) != model.predict_one(x_h):      
                d = cdist(query_instance, pd.DataFrame([x_h]), metric='euclidean').flatten()
                if d[0] < min_d:
                    min_d = d[0]
                    x_ = x_h   
    if x_:
            cfs = x_
            def_sparsity = np.sum(x != cfs)
            return cfs, elpased_time,def_sparsity    

        
                


def add_iteration(df, iteration, new_instance, new_cf, cf_method, model):
    updated_rows = []

    if cf_method == "DICE":
        print('Aggiornamento')

        # Seleziona righe precedenti all'attuale iterazione
        previous_rows = df[df["iteration"] != iteration]

        # Prendi solo l'ultima occorrenza per ogni controfattuale
        last_rows = previous_rows.drop_duplicates(subset=["counterfactual"], keep="last")

        for _, row in last_rows.iterrows():
            updated_validity = model.predict_one(row["counterfactual"])
            if row["validity"] != updated_validity:
                updated_rows.append({
                    "iteration": iteration,
                    "original_instance": row["original_instance"],
                    "counterfactual": row["counterfactual"],
                    "validity": updated_validity,
                    "cf_method": row["cf_method"]
                })

    print('Fine Aggiornamento')

    # Nuova riga per l'istanza corrente
    new_row = {
        "iteration": iteration,
        "original_instance": new_instance,
        "counterfactual": new_cf,
        "validity": model.predict_one(new_cf),
        "cf_method": cf_method
    }

    # Ritorna DataFrame aggiornato
    return pd.concat([df, pd.DataFrame(updated_rows + [new_row])], ignore_index=True)
        
         



    

    

def get_counterexamples_proto(x, df_log, model, target, cats):
    # Disabilita eager execution all'inizio
    tf.compat.v1.disable_eager_execution()
    
    # Ottimizzazione: usa categorie solo se necessarie
    category_values = {
        df_log.columns.get_loc(col): df_log[col].nunique()
        for col in cats
    } if cats else {}
    
    # Ottimizzazione: seleziona solo le colonne necessarie
    cols_needed = [c for c in df_log.columns if c != target]
    X = df_log[cols_needed]
    
    # Ottimizzazione: usa array numpy invece di DataFrame dove possibile
    X_v = X.to_numpy(dtype=np.float32) 
    query_values = pd.DataFrame([x]).values.astype(np.float32)
    
    # Inizializza il modello wrapper
    feature_names = cols_needed
    river_model_wrapper = RiverModelWrapper(model, target, feature_names)
    
    # Calcola i range delle feature in modo efficiente
    X_min = X_v.min(axis=0)
    X_max = X_v.max(axis=0)
    feature_ranges = (X_min.reshape(1, -1), X_max.reshape(1, -1))
    
    # Funzione di predizione
    predict_fn = lambda x: river_model_wrapper.predict_proba(x)
    
    # Parametri di configurazione
    configs = [
        {'theta': 10.0, 'beta': 0.01, 'lr': 0.01, 'threshold': 0.0, 'c_init': 1.0},
        {'theta': 5.0, 'beta': 0.005, 'lr': 0.01, 'threshold': 0.01, 'c_init': 1.0},
        {'theta': 2.5, 'beta': 0.0025, 'lr': 0.05, 'threshold': 0.005, 'c_init': 1.0},
        {'theta': 1.25, 'beta': 0.00125, 'lr': 0.25, 'threshold': 0.0025, 'c_init': 1.0},
        {'theta': 0.625, 'beta': 0.000625, 'lr': 1.25, 'threshold': 0.001, 'c_init': 100.0}
    ]
    
    start_time = time.time()
    
    for attempt, config in enumerate(configs):
        print(f"Tentativo {attempt+1} con parametri: {config}")
        
        try:
            cf = CounterfactualProto(
                predict_fn,
                query_values.shape,
                beta=config['beta'],
                use_kdtree=True,
                cat_vars=category_values,
                ohe=False,
                theta=config['theta'],
                learning_rate_init=config['lr'],
                max_iterations=800,
                c_init=config['c_init'],
                c_steps=10,
                feature_range=feature_ranges
            )
            
            cf.fit(X_v)
            explanation = cf.explain(query_values, threshold=config['threshold'])
            
            if explanation.cf['X'] is not None:
                elapsed_time = time.time() - start_time
                cf_expl = explanation.cf['X'].flatten()
                def_sparsity = np.sum(query_values != explanation.cf['X'])
                
                # Pulizia memoria prima di restituire i risultati
                del cf, explanation
                
                return dict(zip(feature_names, cf_expl)), elapsed_time, def_sparsity
                
        except Exception as e:
            print(f"Errore al tentativo {attempt+1}: {str(e)}")
            continue
    
    # Se nessun controfattuale trovato, cerca il punto più vicino
    elapsed_time = time.time() - start_time
    min_d = float('inf')
    x_ = None
    
    # Ottimizzazione: usa array numpy per calcolare le distanze
    x_array = np.array([x[col] for col in cols_needed])
    
    for x_h in X.to_dict('records'):
        if model.predict_one(x) != model.predict_one(x_h):
            x_h_array = np.array([x_h[col] for col in cols_needed])
            d = np.linalg.norm(x_array - x_h_array)
            if d < min_d:
                min_d = d
                x_ = x_h
    
    if x_:
        sparsity = sum(x[col] != x_[col] for col in cols_needed)
        elapsed_time = time.time() - start_time
        return x_, elapsed_time, sparsity
    
    return None, elapsed_time, 0   
    









    
    
   
   

    
    
    
    

    

def generate_counter_rules(model,inst,df,target):
    feature_names = df.columns.tolist()
    feature_names = [f for f in feature_names if f != target]
    #print(feature_names)
    river_model_wrapper = RiverModelWrapper(model,target,feature_names)
    bbox = sklearn_classifier_wrapper(river_model_wrapper)  
    explainer = LoreTabularExplainer(bbox)
    config = {'neigh_type':'rndgen', 'size':1000, 'ocr':0.1, 'ngen':10}
    
    explainer.fit(df, target, config)
    query_instance = pd.DataFrame([inst])
    query = np.array(query_instance).flatten()
    #print(query)
    #query_instance = query_instance.values.reshape((1, 1, -1))
    exp = explainer.explain(query)
    counter_rules = exp.getCounterfactualRules()
    pred_x = model.predict_one(inst)
    query_features = [col for col in df.columns if col != target]
    query_values = np.array([inst[feat] for feat in query_features if feat in inst], dtype=np.float64).reshape(1, -1)
    if counter_rules == []:
        min_d = float('inf')
        x_ = None
        for _, row in df.iterrows():
            row_values = np.array([row[feat] for feat in query_features if feat in inst], dtype=np.float64).reshape(1, -1)
            if pred_x != model.predict_one(row.to_dict()):
                distances = cdist(query_values,row_values , 'euclidean')[0]
                if distances < min_d:
                    min_d = distances
                x_ = row
        x_ = x_.drop(target)       
        counter_rules = generate_rules(model,x_,df,target)
            
    return counter_rules

def generate_rules(model,inst,df,target):
    feature_names = df.columns.tolist()
    feature_names = [f for f in feature_names if f != target]
    #print(feature_names)
    river_model_wrapper = RiverModelWrapper(model,target,feature_names)
    bbox = sklearn_classifier_wrapper(river_model_wrapper)  
    explainer = LoreTabularExplainer(bbox)

    config = {'neigh_type':'rndgen', 'size':1000, 'ocr':0.1, 'ngen':10}
    
    explainer.fit(df, target, config)
    query_instance = pd.DataFrame([inst])
    query = np.array(query_instance).flatten()
    #print(query)
    #query_instance = query_instance.values.reshape((1, 1, -1))
    exp = explainer.explain(query)
    rules = exp.getRules()
    return rules

   
def generate_similar(model,inst,df,target):
    feature_names = df.columns.tolist()
    feature_names = [f for f in feature_names if f != target]
    #print(feature_names)
    river_model_wrapper = RiverModelWrapper(model,target,feature_names)
    bbox = sklearn_classifier_wrapper(river_model_wrapper)  
    explainer = LoreTabularExplainer(bbox)

    config = {'neigh_type':'rndgen', 'size':1000, 'ocr':0.1, 'ngen':10}
    
    explainer.fit(df, target, config)
    query_instance = pd.DataFrame([inst])
    query = np.array(query_instance).flatten()
    #print(query)
    #query_instance = query_instance.values.reshape((1, 1, -1))
    exp = explainer.explain(query)
    similars = exp.getExemplars()
    dicts = []
    for array in similars:
        #Converti i numeri in formato più leggibile
        #valori_convertiti = [converti_numero(num) for num in array]
        dizionario = dict(zip(feature_names, array))
        dicts.append(dizionario)
    df_similars = pd.DataFrame(dicts)    
    return df_similars,dicts    

def calculate_lore_fidelity(model,inst,df,target):
    feature_names = df.columns.tolist()
    feature_names = [f for f in feature_names if f != target]
    #print(feature_names)
    river_model_wrapper = RiverModelWrapper(model,target,feature_names)
    bbox = sklearn_classifier_wrapper(river_model_wrapper)  
    explainer = LoreTabularExplainer(bbox)

    config = {'neigh_type':'rndgen', 'size':1000, 'ocr':0.1, 'ngen':10}
    
    explainer.fit(df, target, config)
    query_instance = pd.DataFrame([inst])
    query = np.array(query_instance).flatten()
    #print(query)
    #query_instance = query_instance.values.reshape((1, 1, -1))
    exp = explainer.explain(query)
    fidelity = exp.getFidelity()
    print('Fidelity :',fidelity)
   
    return fidelity
    



from sklearn.metrics import mean_squared_error
def check_loss(scaler,ae,X, x_new):
   # Dopo il training:
    x_new = pd.DataFrame([x_new]).values 
    X_train_scaled = scaler.transform(X)
    X_train_reconstructed = ae.predict(X_train_scaled)
    train_losses = np.mean((X_train_scaled - X_train_reconstructed)**2, axis=1)
    mean_loss = np.mean(train_losses)
    std_loss = np.std(train_losses)
    x_scaled = scaler.transform(x_new)
    x_reconstructed = ae.predict(x_scaled)
    x_loss = np.mean((x_scaled - x_reconstructed)**2)
    # confronto
    check = True
    if x_loss > mean_loss + 3 * std_loss:
        print(" Loss anomala: fuori soglia (potenzialmente fuori distribuzione)")
        check = False 
    else:
        print(" Loss nella norma")
   

    return check


def get_percentage_and_df(df_train,processed,target):
    
    processed_c = processed.copy()
    rows = []
    #query_instance = pd.DataFrame([x])
    df_log  = pd.DataFrame()
    #print('Processed_C',processed_c)
    if len(processed_c)>0:#Trasformo lo storico in un DataFrame se ci son """o degli esempi visti """ """ """
        for entry in processed_c.values():
                row = entry['dict_form'].copy()
                row[target] = entry['decision']
                rows.append(row)
        df_proc = pd.DataFrame(rows)
        df_log = pd.concat([df_train,df_proc], ignore_index=True)
    else:
        df_log= df_train
    percentage_dict = (df_log[target].value_counts(normalize=True) * 100).round(2).to_dict()
    feature_names = [f for f in df_log.columns.tolist() if f !=target]
    return  percentage_dict,df_log    

    
    

    
        




from scipy.spatial.distance import cdist

def calculate_distances(x_dict, examples, feature_ranges=None):
    """
    Calcola la distanza euclidea tra x e una lista di dizionari o un DataFrame.
    
    """
    # Estrai feature numeriche da x_dict
    numeric_features = [
        feat for feat, val in x_dict.items() 
        if isinstance(val, (int, float, np.number))
    ]
    if not numeric_features:
        raise ValueError("Nessuna feature numerica trovata in x_dict.")

    # Prepara array di x (solo feature numeriche)
    
    x_values = np.array([x_dict[feat] for feat in numeric_features], dtype=np.float64).reshape(1, -1)
    if len(examples)>0:
        # Gestione input: DataFrame o lista di dizionari
        if isinstance(examples, pd.DataFrame):
        # DataFrame -> estrai solo colonne numeriche corrispondenti a x_dict
            examples_numeric = examples[numeric_features].astype(np.float64)
            valid_examples = examples.to_dict('records')
            examples_array = examples_numeric.values
        else:
        # Lista di dizionari -> filtra valori numerici
            valid_examples = []
            examples_values = []
        
            for example in examples:
                ex_values = []
                valid = True
                for feat in numeric_features:
                    val = example.get(feat)
                    if isinstance(val, (int, float, np.number)):
                        ex_values.append(val)
                    else:
                        try:
                             ex_values.append(float(val))
                        except (ValueError, TypeError):
                            valid = False
                            break
            
                if valid:
                    examples_values.append(ex_values)
                    valid_examples.append(example)
        
            
            examples_array = np.array(examples_values, dtype=np.float64)

    # Normalizzazione 
        if feature_ranges:
            ranges = np.array([feature_ranges.get(feat, 1.0) for feat in numeric_features])
            x_values = x_values / ranges
            examples_array = examples_array / ranges

    # Calcola distanze con cdist
       
        distances = cdist(x_values, examples_array, 'euclidean')[0]
       

    # Combina risultati
        
        results = list(zip(valid_examples, distances))
    
        results = [item for item in results if item[1]if item[1]!=0.0] 
        results.sort(key=lambda item: item[1])
       
    else:
       
        results = []
    return results

def incremental_ae(scaler,ae,X,BATCH_SIZE,EPOCHS):
    
     ae.fit(
       scaler.transform(X),
       scaler.transform(X),
        #X_train,
        #X_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=False)
     return ae,scaler
def incremental_hae(cats,df_log,HIDDEN_DIM,LATENT_DIM,EPOCHS):
    if cats != []:
        X = df_log.drop('target',axis=1)
        feature_names = df_log.columns.tolist()
        category_map =  {}
        for i in range(len(feature_names)):
            if feature_names[i]  in cats:
                    category_map[i] = df_log[feature_names[i]].unique().tolist()
       

        # Separate columns in numerical and categorical.
        categorical_names = [feature_names[i] for i in category_map.keys()]
        categorical_ids = list(category_map.keys())
        numerical_names = [name for i, name in enumerate(feature_names) if i not in category_map.keys()]
        numerical_ids = [i for i in range(len(feature_names)) if i not in category_map.keys()]
        heae_preprocessor, heae_inv_preprocessor = get_he_preprocessor(X=X,
                                                               feature_names=feature_names,
                                                               category_map=category_map)
        trainset_input = heae_preprocessor(X).astype(np.float32)
        trainset_outputs = {
        "output_1": trainset_input[:, :len(numerical_ids)]
        }
        for i, cat_id in enumerate(categorical_ids):
            trainset_outputs.update({
            f"output_{i+2}": X[:, cat_id]
            })
        
        trainset = tf.data.Dataset.from_tensor_slices((trainset_input, trainset_outputs))
        trainset = trainset.shuffle(1024).batch(HIDDEN_DIM  , drop_remainder=True)
        # Define output dimensions.
        OUTPUT_DIMS = [len(numerical_ids)]
        OUTPUT_DIMS += [len(category_map[cat_id]) for cat_id in categorical_ids]

        # Define the heterogeneous auto-encoder.
        heae = HeAE(encoder=ADULTEncoder(hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM),
            decoder=ADULTDecoder(hidden_dim=HIDDEN_DIM, output_dims=OUTPUT_DIMS))
         # Define loss functions.
        he_loss = [keras.losses.MeanSquaredError()]
        he_loss_weights = [1.]

        # Add categorical losses.
        for i in range(len(categorical_names)):
            he_loss.append(keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            he_loss_weights.append(1./len(categorical_names))

        # Define metrics.
        metrics = {}
        for i, cat_name in enumerate(categorical_names):
            metrics.update({f"output_{i+2}": keras.metrics.SparseCategoricalAccuracy()})

        # Compile model.
        heae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
             loss=he_loss,
             loss_weights=he_loss_weights,
             metrics=metrics)
    
    
    
        heae.fit(trainset, epochs=EPOCHS)
def calculate_sparsity(x_dict, cf_x_dict):
    key = list(x_dict.keys())
    x = np.array([x_dict[k] for k in key])
    cf_x = np.array([cf_x_dict[k] for k in key])
    sparsity = np.sum(x != cf_x)
    return sparsity

def filter_dict_by_counter_rules(dict_list, counter_rules, model_balance, x):
   
    matching_examples = []
    pred_x = model_balance.predict_one(x)  # Predizione di x
    if  isinstance(counter_rules,list):
        for example in dict_list:
        # Verifica se soddisfa ALMENO UNA regola (OR)
            satisfies_any_rule = any(
            all(  # Tutte le condizioni della singola regola devono essere vere (AND)
                (example.get(cond['att'], None) > cond['thr'] if cond['op'] == '>' else 
                 example.get(cond['att'], None) <= cond['thr'])
                for cond in rule['premise']
            )
                for rule in counter_rules
            )
        
            if not satisfies_any_rule:
                continue  # Salta se nessuna regola è soddisfatta
        
        # Verifica se la predizione è opposta a x
            pred_example = model_balance.predict_one(example)
            if pred_example != pred_x:
                matching_examples.append(example)
        print(matching_examples)
        return matching_examples
    else:
        check = True
        print(filter_dict_by_rule(dict_list,counter_rules,x,model_balance,check))
        return filter_dict_by_rule(dict_list,counter_rules,x,model_balance,check)
def filter_df_by_counter_rules(df, counter_rules, model_balance, x):
   
    pred_x = model_balance.predict_one(x)
    mask = pd.Series(False, index=df.index)  # Inizializza tutto False
    if isinstance(counter_rules, list):    
    # Applica OR tra le regole
        for rule in counter_rules:
            rule_mask = pd.Series(True, index=df.index)  # Inizializza tutto True per questa regola
        
        # Applica AND tra le condizioni della singola regola
            for cond in rule['premise']:
                att = cond['att']
                op = cond['op']
                thr = cond['thr']
            
                if op == '>':
                    rule_mask &= (df[att] > thr)
                elif op == '<=':
                    rule_mask &= (df[att] <= thr)
        
            mask |= rule_mask  # OR logico tra regole
    
    # Filtra per predizione opposta
        pred_mask = df.apply(
            lambda row: model_balance.predict_one(row.to_dict()) != pred_x,
            axis=1
        )
    
        return df[mask & pred_mask]  # Combina maschere
    else:
        check=False
        print(counter_rules)
        return filter_df_by_rule(df,counter_rules,model_balance,x,check)

def filter_dict_by_rule(dict_list, rule,x,model,check=False):
   
    matching_examples = []
    
    for example in dict_list:
        satisfies_all_conditions = all(
            (example.get(cond['att'], None) > cond['thr'] if cond['op'] == '>' else 
             example.get(cond['att'], None) <= cond['thr'])
            for cond in rule['premise']
        )
        
        if satisfies_all_conditions:
            pred_x = model.predict_one(x)
            pred_example = model.predict_one(example)
            if pred_example == pred_x and not check:
                # Se la predizione è uguale, aggiungi l'esempio alla lista  
                matching_examples.append(example)
            elif check:
                matching_examples.append(example)
                
    
    return matching_examples

def filter_df_by_rule(df, rule, model, x,check=False):
    
    # 1. Filtra per la regola (AND tra condizioni)
    mask = pd.Series(True, index=df.index)
    for cond in rule['premise']:
        att = cond['att']
        op = cond['op']
        thr = cond['thr']
        
        if op == '>':
            mask &= (df[att] > thr)
        elif op == '<=':
            mask &= (df[att] <= thr)
    
    filtered_by_rule = df[mask].copy()  # DataFrame già filtrato per la regola
    
    # 2. Filtra per la predizione uguale a quella di x
    pred_x = model.predict_one(x)
    mask_pred = filtered_by_rule.apply(
        lambda row: model.predict_one(row.to_dict()) == pred_x if not check else model.predict_one(row.to_dict()) != pred_x ,
        axis=1
    )
    
    return filtered_by_rule[mask_pred]  # Applica la maschera sul DataFrame già filtrato


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj


import multiprocessing as mp


def run_cf_fit(x, model, target, feature_names, return_dict):
    river_model_wrapper = RiverModelWrapper(model, target, feature_names)
    query_instance = pd.DataFrame([x])
    CF = cf.CounterfactualExplanation(query_instance.values, river_model_wrapper.predict, method='GS')
    
    CF.fit(n_in_layer=100, first_radius=1.5, dicrease_radius=1.5, sparse=False, verbose=False)
    cf_x = CF.enemy
    return_dict['cf'] = cf_x

def get_counterfactuals_GS(x, df_log, model, target, timeout_seconds=300):
    query_instance = pd.DataFrame([x])
    feature_names = list(query_instance.columns)
    manager = mp.Manager()
    return_dict = manager.dict()
    
    process = mp.Process(target=run_cf_fit, args=(x, model, target, feature_names, return_dict))
    start_time = time.time()
    process.start()
    process.join(timeout_seconds)

    elapsed_time = time.time() - start_time

    if process.is_alive():
        process.terminate()
        process.join()
        print(f"Timeout! Il calcolo ha superato {timeout_seconds} secondi, fermato.")
        # Fallback manuale
        X = df_log.loc[:, df_log.columns != target]
        X = list(X.to_dict(orient='index').values())
        min_d = float('inf')
        x_ = None
        for x_h in X:
            if model.predict_one(x) != model.predict_one(x_h):
                d = cdist(query_instance, pd.DataFrame([x_h]), metric='euclidean').flatten()
                if d[0] < min_d:
                    min_d = d[0]
                    x_ = x_h
        if x_:
            sparsity = np.sum(x != x_)
            return x_, elapsed_time, sparsity
        return None, elapsed_time, None

    if 'cf' in return_dict:
        cf_x = return_dict['cf']
        sparsity = np.sum(x != cf_x)
        x_d = dict(zip(feature_names, cf_x))
        return x_d, elapsed_time, sparsity
    else:
        return None, elapsed_time, None

def convert_dict_list_to_float32(dict_list):
    return [
        {
            k: np.float32(v) if isinstance(v, float) else v
            for k, v in d.items()
        }
        for d in dict_list
    ]


def create_log(processed_c,target):
    rows = []
    for entry in processed_c.values():
        row = entry['dict_form'].copy()
        row[target] = entry['decision']
        rows.append(row)
    df_proc = pd.DataFrame(rows)
    return df_proc