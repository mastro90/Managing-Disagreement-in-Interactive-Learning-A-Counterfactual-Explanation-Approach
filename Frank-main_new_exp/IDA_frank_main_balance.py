# Importing required libraries and modules
from river import rules, tree, datasets, drift, metrics, evaluate
from IPython import display
import random
import functools
from itertools import combinations, product
import numpy as np
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import fatf.utils.data.datasets as fatf_datasets
import fatf.fairness.data.measures as fatf_dfm
import fatf.utils.data.tools as fatf_data_tools
from tqdm import tqdm
from matplotlib import pyplot as plt
from classes import *  # Importing custom classes
from frank_algo import *  # Importing custom algorithms
import pickle
from sklearn.metrics import classification_report
import re
from river import imblearn
from river import preprocessing
from river import optim
import os 
from river import ensemble
from sklearn.preprocessing import StandardScaler
import json
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import copy
import orjson



class Frank_balance:
    """
    Class representing the Frank decision-making process.

    Attributes:
        RULE (bool): Flag indicating whether predefined rules are applied.
        PAST (bool): Flag indicating whether past decisions are considered.
        SKEPT (bool): Flag indicating whether skepticism is applied.
        GROUP (bool): Flag indicating whether group fairness checks are performed.
        EVA (bool): Flag indicating whether evaluation is performed.
        n_bins (int): Number of bins for XAI (syntethic records).
        n_var (int): Number of variables for XAI (syntethic records).
        maxc (int): Maximum number of modifief features for XAI (syntethic records).
        simulated_user (object): Simulated user model.
        train_check (bool): Flag indicating whether the model is pre-trained.
        Y (list): Target labels of data streams.
        X (list): Feature vectors of data streams.
        X_test (list): For eva.
        Y_test (list): For eva.

        attr_list (list): List of attribute names.
        fairness_records (list): Records for fairness evaluation.
        protected (list): List of protected (sensitive) attributes.
        protected_values (numpy.ndarray): Unique values of protected attributes.
        cats (list): List of categorical attributes.
        rule_att (str): Attribute for predefined rules.
        rule_value (bool): Value for predefined rules.
        model (object): Incremental machine learning model.

    Methods:
        __init__: Initializes the Frank object.
        train: Trains the model.
        start: Starts the decision-making process.
    """

    def __init__(self, name,model,user_model, df, target, protected, cats, rule_att, rule_value, RULE, PAST, SKEPT, GROUP, X_test,
                 Y_test, EVA, n_bins, n_var, maxc,X_train,Y_train,user):
        """
        Initializes the Frank object.

        Args:
            user_model (object): Simulated user model.
            df (DataFrame): Input DataFrame.
            target (str): Target column name.
            protected (list): List of protected attributes.
            cats (list): List of categorical attributes.
            rule_att (str): Attribute for predefined rules.
            rule_value (bool): Value for predefined rules.
            RULE (bool): Flag indicating whether Ideal Rule Check is performed.
            PAST (bool): Flag indicating whether Individual Fairness Check is performed..
            SKEPT (bool): Flag indicating whether Skeptical Learning Check is performed.
            GROUP (bool): Flag indicating whether Group Fairness Check are performed.
            X_test (list): Test feature vectors.
            Y_test (list): Test target labels.
            EVA (bool): Flag indicating whether evaluation is performed.
            EVA (bool): Flag indicating whether evaluation is performed.
            n_bins (int): Number of bins for XAI (syntethic records).
            n_var (int): Number of variables for XAI (syntethic records).

        Returns:
            None     
        """

        # Setting attributes
        self.name = name
        self.X_train = X_train
        self.Y_train = Y_train
        self.df = df
        self.RULE = RULE
        self.PAST = PAST
        self.SKEPT = SKEPT
        self.GROUP = GROUP
        self.EVA = EVA
        self.n_bins = n_bins
        self.n_var = n_var
        self.maxc = maxc
        self.simulated_user = user_model
        self.train_check = False
        self.Y = list(df[target])
        self.Y = [bool(y) for y in self.Y]
        self.X = df.loc[:, df.columns != target]
        self.X = list(self.X.to_dict(orient='index').values())
        self.X_test = X_test
        self.Y_test = Y_test
        self.target = target
        self.attr_list = list(df.columns)
        self.fairness_records = [len(self.X) - 1]
        for i in range(0, 100, 5)[1:]:
            self.fairness_records.append(percentage(i, len(self.X)))
        self.protected = protected
        self.protected_values = df[protected[0]].unique()
        self.cats = cats
        self.rule_att = rule_att
        self.rule_value = rule_value
        self.user = user
        self.model = model
        self.initial_model = pickle.loads(pickle.dumps(self.model))
        self.processed = dict()
        self.evaluation_results = []
        self.stats = dict()
        self.stats[False] = dict()
        self.stats[True] = dict()
        for e in ['user', 'machine']:
            self.stats[False][e] = dict()
            self.stats[True][e] = dict()
            self.stats[False][e]['tried'] = 0
            self.stats[True][e]['tried'] = 0
            self.stats[False][e]['got'] = 0
            self.stats[True][e]['got'] = 0
            if e == 'user':
                self.stats[False][e]['conf'] = 1
                self.stats[True][e]['conf'] = 1
            else:
                self.stats[False][e]['conf'] = 0
                self.stats[True][e]['conf'] = 0

        #various counters for testing/debugging purposes
        self.rules_count = 0
        self.past_count = 0
        self.ok_count = 0
        self.no_count = 0
        self.xai_ok = 0
        self.xai_no = 0
        self.skept_count = 0
        self.agree_count = 0
        self.disagree_count = 0
        self.acc = metrics.Accuracy()
        self.F1 = metrics.F1()
        self.accs = []
        self.F1s = []

    def train(self, X_frank_train, Y_frank_train):
        """
        Pre-training the model.

        Args:
            X_frank_train (list): Training feature vectors.
            Y_frank_train (list): Training target labels.

        Returns:
            None
        """

        self.X_frank_train = X_frank_train
        self.Y_frank_train = Y_frank_train
        df_train = pd.DataFrame(self.X_frank_train)
        df_train[self.target] = self.Y_frank_train
        percentage_dict,df_log = get_percentage_and_df(df_train,self.processed,self.target)
       
        
        for x, y in zip(self.X_frank_train, self.Y_frank_train):
            
            y_pred = self.model.predict_one(x)
            acc_1 = self.acc.update(y,y_pred)
            self.accs.append(str(copy.deepcopy(acc_1)))
            f1_ = self.F1.update(y,y_pred)
            self.F1s.append(str(copy.deepcopy(f1_)))
            self.model.learn_one(x, y)
        
            

        from river import metrics
        from collections import Counter
        print("Model trained")

        accuracy = metrics.Accuracy()
        predictions = []
       
        
        

        for x, y in zip(self.X_frank_train, self.Y_frank_train):
            
           
            y_pred = self.model.predict_one(x)
            accuracy.update(y, y_pred)
            predictions.append(y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Distribution of predictions: {Counter(predictions)}")

            
            
            

        self.train_check = True
        print("Model trained")
        return self
    



    def start(self):
        """
        Starts the Hybrid Decision-Making process.

        Returns:
            processed (dict): Processed records.
            evaluation_results (list): Evaluation results.
        """
        
        
       
        
        print('Train')

        #frank_fairness, frank_acc = evaluation_frank(self.X_test, self.Y_test, self.model, self.protected)
        
        predictions = []#Lista usata per salvare le predictions a ogni iterazione , utile per il classification_report
        accuracy_score = []#Lista usata per salvare i vari punteggi di accuracy
        f1_score = []#Lista salvata per salvare i vari punteggi di f1
        rules = []
        counter_rules = []
        frank_cms = []
        equality = []#Lista usata per verificare quando cambiano i modelli
        df_train = pd.DataFrame(self.X_frank_train)
        df_train[self.target] = self.Y_frank_train
        percentage_dict = (df_train[self.target].value_counts(normalize=True) * 100).round(2).to_dict()
        X = df_train.drop(columns=[self.target]).values
        Y = df_train[self.target].values
        times_RL = []
        times_gs = []
        times_LORE = []
        proximities_RL = []
        proximities_LORE = []
        proximities_gs = []
        sparsities_RL = []
        sparsities_gs = []
        sparsities_LORE = []
        plausabilities_gs = []
        plausabilities_LORE = []
        plausabilities_RL = []
        sparsities_DICE = []
        proximities_DICE = []
        plausabilities_DICE = []
        skepticisms = []
        times_DICE = []
       # DataFrame iniziale
        df_validity = pd.DataFrame(columns=[
        "iteration",            # Iterazione corrente (timestamp logico)
        "original_instance",    # Dati dell'istanza originale
        "counterfactual",       # Dati del controfattuale
        "validity",             # True / False 
        "cf_method"             # Metodo usato per generare il controfattuale
        ])
        self.X = convert_dict_list_to_float32(self.X)
        df_log = pd.DataFrame()
        for i in tqdm(range(len(self.X))):
           
            
        


            #copied_model = pickle.loads(pickle.dumps(model))
           
            

        
        

            relabel = False #When this is set to True, Re-Labelling is triggered
            x = self.X[i]
            y = self.Y[i]
            #print(x)
            percentage_dict,df_log = get_percentage_and_df(df_train,self.processed,self.target)
            print(percentage_dict)
            
            
            
              
            record = tuple(list(x.values()))
            record_user = np.array(list(x.values())).reshape(1, -1)
            
            user_truth = self.simulated_user.predict(np.array(record).reshape(1, -1)[0], y)
            prediction = self.model.predict_one(x)

            
            predictions.append(prediction) 
           
            

            if record in list(self.processed.keys()): #Duplicated record
                self.processed[record]['times'] += 1
                print("Record already processed...")
                old_decision = self.processed[record]['decision']

                if user_truth == old_decision:
                    print("And you are consistent! Decision accepted.")
                    decision = old_decision

                else:
                    print("Inconsistent. You previously said:", old_decision, "Want to change old decision?")

                    confirm = random.choices(population=[False, True], weights=[0.8, 0.2], k=1)[0]

                    if confirm == False:
                        decision = old_decision
                    else:
                        decision = user_truth
                        relabel = True

                self.stats[user_truth]['user']['tried'] += 1
                self.stats[prediction]['machine']['tried'] += 1

                if decision == user_truth:
                    self.stats[user_truth]['user']['got'] += 1
                if decision == prediction:
                    self.stats[prediction]['machine']['got'] += 1

            else:

                self.processed[record] = dict()
                self.processed[record]['notes'] = []
                self.processed[record]['vs'] = None
                self.processed[record]['ideal'] = None
                self.processed[record]['times'] = 1

                try:
                    pred_proba = self.model.predict_proba_one(x)[prediction]
                except:
                    pred_proba = 0

                try:
                    user_proba = self.model.predict_proba_one(x)[user_truth]
                except:
                    print("Still unlearned...")
                    user_proba = 1

                #Skeptical Learning parameters:
                user_confidence = self.stats[user_truth]['user']['conf']
                mach_confidence = self.stats[prediction]['machine']['conf']

                self.stats[user_truth]['user']['tried'] += 1
                self.stats[prediction]['machine']['tried'] += 1

                ideal_value = ideal_record_test(x, self.rule_att, self.rule_value) #Is record covered by Ideal Rule Check?

                vs_records, vs_decision = get_value_swap_records(x, self.processed,
                                                                 self.protected, self.attr_list) #Is record covered by Individual Fairness Check?

                if user_truth == prediction:
                    skepticism = 0
                else:
                    skepticism = mach_confidence * pred_proba - user_confidence * user_proba

                if ideal_value is not None and user_truth != ideal_value and self.RULE: #User is consistent w.r.t. Ideal Rule
                    self.rules_count += 1
                    decision = ideal_value
                    self.processed[record]['ideal'] = False
                    if prediction == ideal_value:
                        self.stats[prediction]['machine']['got'] += 1

                elif ideal_value is not None and user_truth == ideal_value and self.RULE: #User is not consistent w.r.t. Ideal Rule
                    decision = ideal_value
                    self.processed[record]['ideal'] = True
                    if prediction == ideal_value:
                        self.stats[prediction]['machine']['got'] += 1

                elif vs_decision is not None and user_truth != vs_decision and self.PAST: #IRC not triggered. User not consistent w.r.t. Individual Fairnesss
                    self.processed[record]['vs'] = True
                    self.past_count += 1
                    for rec in vs_records:
                        self.processed[rec]['vs'] = True
                    confirm = random.choices(population=[False, True], weights=[0.8, 0.2], k=1)[0]
                    if confirm in [0, "0", False]:
                        decision = vs_decision
                        if prediction == vs_decision:
                            self.stats[prediction]['machine']['got'] += 1
                    elif confirm in [1, "1", True]:
                        decision = user_truth
                        self.stats[user_truth]['user']['got'] += 1
                        if prediction == user_truth:
                            self.stats[prediction]['machine']['got'] += 1
                        for rec in vs_records:
                            self.processed[rec]['decision'] = user_truth
                        relabel = True

                elif vs_decision is not None and user_truth == vs_decision and self.PAST: #IRC not triggered. User not consistent w.r.t. Individual Fairnesss
                    self.processed[record]['vs'] = True
                    for rec in vs_records:
                        self.processed[rec]['vs'] = True
                    decision = vs_decision
                    if prediction == vs_decision:
                        self.stats[prediction]['machine']['got'] += 1
                
                else: #Other conditions not triggered. Skeptical Learning Check
                    if user_truth != prediction and self.SKEPT:
                        skepticisms.append({str(i):skepticism})
                        if skepticism > 0.5:
                            #cf_alibi,time_alibi,sparsity_alibi = get_examples_4(x,df_log,self.model,self.target,self.cats)
                            
                            # times_RL.append(time_alibi)
                            # sparsities_RL.append({str(i):sparsity_alibi})
                            # proximity_RL = calculate_distances(x,pd.DataFrame([cf_alibi]),feature_ranges=None)
                            # proximity_alibi = proximity_RL[0][1]
                            # plausability_RL = calculate_distances(cf_alibi,df_log,feature_ranges=None)
                            # proximities_RL.append({str(i):proximity_alibi})
                            # plausability_alibi =  plausability_RL[0][1]
                            # plausabilities_RL.append({str(i): plausability_alibi})
                            # df_validity=add_iteration(df_validity, i, x, cf_alibi, "RL",self.model)
                            df_similars,dicts = generate_similar(self.model,x,df_log,self.target)
                            # print(dicts)
                            # rules = generate_rules(self.model,x,df_log,self.target)
                            # print('Rules:',rules)
                            df_log_1 = df_log.drop(self.target,axis=1)
                            #idx_dataset = filter_df_by_rule(df_log_1, rules,self.model,x)
                            #print('Esempi che rispettano la regola nello stroico : ',idx_dataset)
                            #distances_x_hist_rule = calculate_distances(x, idx_dataset, feature_ranges=None)
                            #print('Distanza tra x e storico rispettante la regola : ',distances_x_hist_rule)
                            #start_time = time.time()
                            counter_rules = generate_counter_rules(self.model,x,df_log,self.target)
                            #print('Contro regole:',counter_rules)
                            #idx_dict_1 = filter_df_by_counter_rules(df_similars, counter_rules,self.model,x)
                            #time_LORE= time.time()-start_time
                            #times_LORE.append(time_LORE)
                            #idx_dataset_1 = filter_df_by_counter_rules(df_log_1, counter_rules,self.model,x)
                            #print('Esempi che rispettano la contro regola nello stroico : ',idx_dataset_1)
                            #distances_x_counter_rules = calculate_distances(x, idx_dataset_1, feature_ranges=None)
                            #print('Distanza tra x e storico rispettante la contro regola : ',distances_x_counter_rules)
                            #idx_dict = filter_dict_by_rule(dicts, rules,x,self.model)
                            #print('Esempi che rispettano la regola negli esempi generati : ',idx_dict)
                            #idx_dataset = filter_df_by_rule(df_similars, rules,self.model,x)
                            #distances_x_generate_rule = calculate_distances(x, idx_dict, feature_ranges=None)
                           #print('Distanza tra x e esempi generati rispettante la regola : ',distances_x_generate_rule)
                            
                            #print('Esempi che rispettano la contro regola negli esempi generati : ',idx_dict_1)
                            idx_dict_1 = filter_dict_by_counter_rules(dicts, counter_rules,self.model,x)
                            #print('Esempi che rispettano la contro regola negli esempi generati : ',idx_dict_1)#Controfattuali di LORE
                            distances_x_generate_counter_rules = calculate_distances(x, idx_dict_1, feature_ranges=None)#Proximity di LORE
                            #print('Distanza tra x e esempi generati rispettante la contro regola : ',distances_x_generate_counter_rules)
                            try:
                                 counterfactual_lore = distances_x_generate_counter_rules[0][0]
                            except:
                                  idx_dataset_1 = filter_df_by_counter_rules(df_log_1, counter_rules,self.model,x)
                                  distances_x_counter_rules = calculate_distances(x, idx_dataset_1, feature_ranges=None)
                                  counterfactual_lore = distances_x_counter_rules[0][0]    
                            #print('Controfattuali LORE:',counterfactual_lore)
                            #sparsity_lore = calculate_sparsity(x,counterfactual_lore)
                        
                            #proximity_lore = distances_x_generate_counter_rules[0][1]

                            # distance_cf_lore_hist =  calculate_distances(counterfactual_lore, df_log, feature_ranges=None)
                            # plausability_lore =  distance_cf_lore_hist[0][1]
                            # print('sparsity_lore : ',sparsity_lore)
                            # print('proximity_lore:',proximity_lore)
                            # print('plausability_lore:',plausability_lore)
                            # sparsities_LORE.append({str(i):sparsity_lore})
                            # proximities_LORE.append({str(i):proximity_lore})
                            # plausabilities_LORE.append({str(i):plausability_lore})
                            # df_validity=add_iteration(df_validity, i, x, counterfactual_lore, "LORE",self.model)
                            # cf_gs,time_gs,sparsity_gs =  get_examples_2(x,df_log,self.model,self.target)#Controfattuali con Growing Sphere
                            # times_gs.append(time_gs)
                            # distance_x_cf_gs=  calculate_distances(cf_gs, pd.DataFrame([x]), feature_ranges=None)
                            # try:
                            #      proximity_gs = distance_x_cf_gs[0][1]
                            # except:
                            #     d = cdist(pd.DataFrame([cf_gs]), pd.DataFrame([x]), metric='euclidean').squeeze()
                            #     proximity_gs = d      
                            # distance_cf_gs_hist = calculate_distances(cf_gs, df_log, feature_ranges=None)
                            # plausability_gs = distance_cf_gs_hist[0][1]
                            #print('Controfattuali gs:',cf_gs)
                            #print('sparsity_gs : ',sparsity_gs)
                            #print('proximity_gs:',proximity_gs)
                            #print('plausability_gs',plausability_gs)
                            # sparsities_gs.append({str(i):sparsity_gs})
                            # proximities_gs.append({str(i):proximity_gs})
                            # plausabilities_gs.append({str(i):plausability_gs})
                            # df_validity=add_iteration(df_validity, i, x, cf_gs, "GS",self.model)
                            #cf_dice,time_dice,sparsity_dice =  get_examples_44(df_log,x,self.model,self.cats,self.target,self.X_frank_train,self.Y_frank_train)#Controfattuali con Growing Sphere
                            # times_DICE.append(time_dice)
                            # distance_x_cf_dice=  calculate_distances(cf_dice, pd.DataFrame([x]), feature_ranges=None)
                            # try:
                            #      proximity_dice = distance_x_cf_dice[0][1]
                            # except:
                            #       d = cdist(pd.DataFrame([cf_dice]), pd.DataFrame([x]), metric='euclidean').squeeze()
                            #       proximity_dice = d      
                            # proximities_DICE.append({str(i):proximity_dice})
                            # distance_cf_dice_hist = calculate_distances(cf_dice, df_log, feature_ranges=None)
                            # plausability_dice = distance_cf_dice_hist[0][1]
                            # print('Controfattuali dice:',cf_dice)
                            # print('sparsity_gs : ',sparsity_gs)
                            # print('proximity_gs:',proximity_gs)
                            # print('plausability_gs',plausability_gs)
                            # sparsities_DICE.append({str(i):sparsity_dice})
                            # plausabilities_DICE.append({str(i):plausability_dice})
                            # df_validity=add_iteration(df_validity, i, x, cf_dice, "DICE",self.model)
                            # print('Inserimento nuova riga')
                            skepticisms.append({str(i):skepticism})
                            # self.skept_count += 1
                            confirm = self.simulated_user.believe()
                            #confirm = None
                          
                            if confirm in [0, "0", False]:
                                # self.no_count += 1
                               
                                decision = user_truth
                                self.stats[user_truth]['user']['got'] += 1
                                df_similars,dicts = generate_similar(self.model,counterfactual_lore,df_log,self.target)
                                Y_h = []
                                new_dicts = []
                                iter = 0
                                print('Utente non ha confermato')
                                for e in range(0,10):
                                    check_dict = ((df_log[list(dicts[e])] == pd.Series(dicts[e])).all(axis=1)).any()#Controllo se un esempio è stato già generato precedentemnete
                                    if not check_dict:       
                                        y_h = self.model.predict_one(dicts[e])
                                        print(user_truth,y_h)
                                        y_pred = self.model.predict_one(x)
                                        acc_1 = self.acc.update(not(user_truth),y_pred)    
                                        f1_ = self.F1.update(not(user_truth),y_pred)
                                        self.accs.append(str(copy.deepcopy(acc_1)))
                                        self.F1s.append(str(copy.deepcopy(f1_)))
                                        self.model.learn_one(dicts[e],not(user_truth))
                                        Y_h.append(not(user_truth))
                                        new_dicts.append(dicts[e])
                                        iter+=1
                                    
                               
                                if Y_h!=[]:   
                                    df_similars = pd.DataFrame(new_dicts)
                                    df_similars[self.target] = Y_h
                                    df_train = pd.concat([df_similars[:10], df_train], ignore_index=True)                                  
                            else:
                                self.ok_count += 1
                                decision = prediction
                                self.stats[prediction]['machine']['got'] += 1
                        else:
                            self.disagree_count += 1
                            decision = user_truth
                            self.stats[user_truth]['user']['got'] += 1
                    else:
                        self.agree_count += 1
                        decision = user_truth
                        self.stats[user_truth]['machine']['got'] += 1
                        self.stats[user_truth]['user']['got'] += 1

                #Once the final decision has been taken, the model is updated. Internal data structure is also updated
               
                X = df_log.drop(columns=[self.target]).values
                #print(self.processed)
                self.processed[record]['dict_form'] = x
                self.processed[record]['decision'] = decision
                self.processed[record]['user'] = user_truth
                self.processed[record]['machine'] = prediction
                
                
            
                
                
                
                try:
                    acc_1 =  self.acc.update(decision,prediction)
                    f1_= self.F1.update(decision,prediction)
                    self.accs.append(str(copy.deepcopy(acc_1)))
                    self.F1s.append(str(copy.deepcopy(f1_)))
                    self.model.learn_one(x, decision)
                except:
                     print('err')
                     print(x,decision) 
                     X =  df_log.drop(columns=[self.target]).to_dict(orient="records")
                     Y = df_log[self.target]
                     self.model = self.initial_model
                     X = convert_dict_list_to_float32(X)
                     for x_train_sample, y_train_sample in zip(X, Y):
                        self.model.learn_one(x_train_sample, y_train_sample)
                     self.model.learn_one(x, decision)    


                _,df_log = get_percentage_and_df(df_train,self.processed,self.target)
                X = df_log.drop(columns=[self.target]).values
                
               
                
                
                
                
                
                
                
                


            try:
                self.stats[user_truth]['user']['conf'] = self.stats[user_truth]['user']['got'] / self.stats[user_truth]['user']['tried']
            except:
                self.stats[user_truth]['user']['conf'] = 1

            try:
                self.stats[prediction]['machine']['conf'] = self.stats[prediction]['machine']['got'] / self.stats[prediction]['machine']['tried']
            except:
                self.stats[prediction]['machine']['conf'] = 0

            if relabel == True:
               
                self.model = self.initial_model
                X =df_log.drop(columns=[self.target]).to_dict(orient="records")
                Y = df_log[self.target]




                for x_train_sample, y_train_sample in zip(X, Y):
                     self.model.learn_one(x_train_sample, y_train_sample)
                   
                  

               

                percentage_dict,_ = get_percentage_and_df(df_train,self.processed,self.target)
                print(percentage_dict)
             
                

                   
                   
                   
                   
                   
                   
            
           
           
                


            
            if i in self.fairness_records and self.GROUP:
                DN, PP, ext = get_fairness(self.model, self.protected, self.processed, self.protected_values)
                fairnes_relabel = DN[:round(len(DN) * 0.25)] + PP[:round(len(PP) * 0.25)]
                for e in fairnes_relabel:
                    self.processed[e[1]]['decision'] = not self.processed[e[1]]['decision']
                
                self.model = self.initial_model

                for x_train_sample, y_train_sample in zip(self.X_frank_train, self.Y_frank_train):
                    self.model.learn_one(x_train_sample, y_train_sample)
                   

                for proc in (self.processed.keys()):
                   
                   
                    x_relabel = self.processed[proc]['dict_form']
                    y_relabel = self.processed[proc]['decision']
                    self.model.learn_one(x_relabel, y_relabel)
                    
                percentage_dict,_ = get_percentage_and_df(df_train,self.processed,self.target)
                
               
        
                

                
            
                
           
            
               

          
            
            
            
         
            if self.EVA:
                human_fairness, human_acc, systemic = evaluation_human(self.processed, self.protected, self.Y,
                                                                       self.attr_list)
               
                frank_fairness, frank_acc,frank_f1,frank_cm = evaluation_frank(self.X_test, self.Y_test, self.model, self.protected)
                accuracy_score.append(frank_acc)
                f1_score.append(frank_f1)
                self.evaluation_results.append([human_fairness, human_acc, systemic, frank_fairness, frank_acc,
                                                self.rules_count, self.past_count,
                                                self.ok_count, self.no_count,
                                                self.xai_ok, self.xai_no,
                                                self.skept_count, self.agree_count, self.disagree_count
                                                ])


                

            

          
            frank_acc,frank_f1,frank_cm = calculate_metrics(self.X_test, self.Y_test, self.model)
            accuracy_score.append(frank_acc)
            f1_score.append(frank_f1)
            frank_cms.append(frank_cm)



            
            



           
            if i ==  (len(self.X)-1):
               

                dir = "results"+"__"+self.name
                os.makedirs(dir, exist_ok=True)
                
                metrics_gs = {
                #"time_steps": time_gs,
                "sparsity": sparsities_gs,
                "proximity": proximities_gs,
                "plausability":plausabilities_gs,
                 "method": "GS", 
                "dataset": self.name
                }
                metrics_lore = {
                #"time_steps": time_lore,
                "sparsity": sparsities_LORE,
                "proximity": proximities_LORE,
                 "plausability":plausabilities_LORE,
                 "method": "LORE",  
                "dataset": self.name
                }
                metrics_RL = {
                #"time_steps": time_lore,
                "sparsity": sparsities_RL,
                "proximity": proximities_RL,
                 "plausability":plausabilities_RL,
                 "method": "RL", 
                "dataset": self.name
                }

                metrics_DICE = {
                "sparsity": sparsities_DICE,
                "proximity": proximities_DICE,
                 "plausability":plausabilities_DICE,
                 "method": "DICE",  
                "dataset": self.name
                }

                skept = {
                    "skept":skepticisms
                }
                with open(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+'model.pkl'), 'wb') as file:
                        pickle.dump(self.model, file)
                with open(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+"Accuracy.txt"), "w") as f:
                        for i, value in enumerate(accuracy_score):
                            f.write(f"{i} {value}\n")
                with open(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+"F1.txt"), "w") as f:
                        for i, value in enumerate(f1_score):
                            f.write(f"{i} {value}\n")
                

                with open(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+'times_gs.txt'), 'w') as f:
                        for i, value in enumerate(times_gs):
                            f.write(f"{i} {value}\n")

                with open(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+'times_RL.txt'), 'w') as f:
                        for i, value in enumerate(times_RL):
                            f.write(f"{i} {value}\n")
                
                with open(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+'times_LORE.txt'), 'w') as f:
                        for i, value in enumerate(times_LORE):
                            f.write(f"{i} {value}\n")
               
                with open(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+"LORE_metrics.json"), "wb") as f:
                     f.write(orjson.dumps(convert_numpy(metrics_lore)))     

                with open(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+"GS_metrics.json"), "wb") as f:
                    f.write(orjson.dumps(convert_numpy(metrics_gs)))         

                with open(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+"RL_metrics.json"), "wb") as f:
                     f.write(orjson.dumps(convert_numpy(metrics_RL)))
                with open(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+"DICE_metrics.json"), "wb") as f:
                     f.write(orjson.dumps(convert_numpy(metrics_DICE)))
                with open(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+"skept.json"), "wb") as f:
                     f.write(orjson.dumps(convert_numpy(skept)))      
                df_validity.to_csv(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+"validity.csv"), index=False)
                for x,y in zip(self.X_test,self.Y_test):
                    y_pred = self.model.predict_one(x)
                    acc_1 = self.acc.update(y,y_pred)    
                    f1_ = self.F1.update(y,y_pred)
                    self.accs.append(str(copy.deepcopy(acc_1)))
                    f1_ = self.F1.update(y,y_pred)
                    self.F1s.append(str(copy.deepcopy(f1_)))
                with open(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+"Acc_prequential.txt"), "w") as f:
                    for i, value in enumerate(self.accs):
                         f.write(f"{i} {value}\n")
                with open(os.path.join(dir,'User_'+str(self.user)+self.name+str(self.model)+"F1_prequential.txt"), "w") as f:
                    for i, value in enumerate(self.F1s):
                        f.write(f"{i} {value}\n")

                









        
            
            
            
        
        
        
        
        
        

        
        
        


            
        
        return self.processed, self.evaluation_results,accuracy_score,f1_score,equality
    


