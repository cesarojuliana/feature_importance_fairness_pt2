import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.datasets import BinaryLabelDataset

import fairness

def standard_scaler(df_train, df_test, columns_num):
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    scaler = StandardScaler().fit(df_train[columns_num])
    df_train_norm = scaler.transform(df_train[columns_num])
    df_train_norm = pd.DataFrame(df_train_norm, columns=columns_num)
    df_train_norm = df_train_norm.join(df_train.drop(columns_num, axis=1))

    df_test_norm = scaler.transform(df_test[columns_num])
    df_test_norm = pd.DataFrame(df_test_norm, columns=columns_num)
    df_test_norm = df_test_norm.join(df_test.drop(columns_num, axis=1))
    return df_train_norm, df_test_norm

def create_df_aif(df_train, df_test, label, protected_attribute, metadata):
    df_train_aif = BinaryLabelDataset(df = df_train, label_names=[label], 
                                      protected_attribute_names = [protected_attribute], 
                                      instance_weights_name=None, unprivileged_protected_attributes=[], 
                                      privileged_protected_attributes=[], metadata=metadata)
    
    df_test_aif = BinaryLabelDataset(df = df_test, label_names=[label], 
                                     protected_attribute_names = [protected_attribute], 
                                     instance_weights_name=None, unprivileged_protected_attributes=[], 
                                     privileged_protected_attributes=[], metadata=metadata)    
    return df_train_aif, df_test_aif

def run(df, protected_attribute, label, columns_num, dataset_name, unprivileged_groups, 
        privileged_groups, metadata):
    df_train, df_test = train_test_split(df, train_size=0.8, shuffle=True)
    print(df_test.shape)
    df_test = df_test[:1000]
    print(df_test.shape)
    df_train, df_test = standard_scaler(df_train, df_test, columns_num)
    df_train = df_train.set_index(protected_attribute).reset_index()
    df_test = df_test.set_index(protected_attribute).reset_index()
    df_train.to_csv('data/{}_train.csv'.format(dataset_name), index=False)
    df_test.to_csv('data/{}_test.csv'.format(dataset_name), index=False)
    X_train, y_train = df_train.drop(label, axis=1), df_train[label]
    X_test, y_test = df_test.drop(label, axis=1), df_test[label]
    df_train_aif, df_test_aif = create_df_aif(df_train, df_test, label, 
                                          protected_attribute, metadata)

    df_train_us_dmin1 = fairness.fairCorrectUnder(df_train, pa=protected_attribute, label=label, fav=1, d=-1)
    df_train_us_dmin1.to_csv('data/{}_train_usd-1.csv'.format(dataset_name), index=False)
    X_train_us_dmin1, y_train_us_dmin1 = df_train_us_dmin1.drop(label, axis=1), df_train_us_dmin1[label]

    df_train_us_d0 = fairness.fairCorrectUnder(df_train, pa=protected_attribute, label=label, fav=1, d=0)
    df_train_us_d0.to_csv('data/{}_train_usd0.csv'.format(dataset_name), index=False)
    X_train_us_d0, y_train_us_d0 = df_train_us_d0.drop(label, axis=1), df_train_us_d0[label]

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    RW.fit(df_train_aif)
    df_train_aif_rw = RW.transform(df_train_aif)
    weights = df_train_aif_rw.instance_weights

    result = []
    dict_models = {'lr': LogisticRegression(),
                   'gb': GradientBoostingClassifier(subsample=0.9),
                   'rf': RandomForestClassifier(max_depth=5, min_samples_leaf=2),
                   'svm': SVC(probability=True)}

    for model_name, model in dict_models.items():

        # Model with bias
        method_name = 'orig'
        model.fit(X_train, y_train)
        res = fairness.compute_metrics(model, X_test, y_test, X_train, y_train, df_test_aif, 
                                          unprivileged_groups, privileged_groups, protected_attribute, False)
        name = '_'.join([dataset_name, model_name, method_name])
        res['name'] = name
        result.append(res)
        pickle.dump(model, open('models/{}.pkl'.format(name), 'wb'))

        # Model with undersampling with d=-1
        method_name = 'usd-1'
        model.fit(X_train_us_dmin1, y_train_us_dmin1)
        res = fairness.compute_metrics(model, X_test, y_test, X_train, y_train, df_test_aif, 
                                       unprivileged_groups, privileged_groups, protected_attribute, False)
        name =  '_'.join([dataset_name, model_name, method_name])
        res['name'] = name
        result.append(res)
        pickle.dump(model, open('models/{}.pkl'.format(name), 'wb'))

        # Model with undersampling with d=0
        method_name = 'usd0'
        model.fit(X_train_us_d0, y_train_us_d0)
        res = fairness.compute_metrics(model, X_test, y_test, X_train, y_train, df_test_aif, 
                                       unprivileged_groups, privileged_groups, protected_attribute, False)
        name = '_'.join([dataset_name, model_name, method_name])
        res['name'] = name
        result.append(res)
        pickle.dump(model, open('models/{}.pkl'.format(name), 'wb'))

        # Model with reweghing
        method_name = 'rw'
        model.fit(X_train, y_train, sample_weight=weights)
        res = fairness.compute_metrics(model, X_test, y_test, X_train, y_train, df_test_aif, 
                                       unprivileged_groups, privileged_groups, protected_attribute, False)
        name = '_'.join([dataset_name, model_name, method_name])
        res['name'] = name
        result.append(res)
        pickle.dump(model, open('models/{}.pkl'.format(name), 'wb'))

    df_result = pd.DataFrame(result)
    df_result = df_result.set_index('name').reset_index()
    df_result.to_csv('data/result_fairness_{}.csv'.format(dataset_name), index=False)
    return df_result
