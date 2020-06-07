# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats
import shap
import math

import warnings
warnings.filterwarnings("ignore")

from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

def consitency(X, y, protect_attribute, n_neighbors=5):
    """Calculate consistency defined in: 
    https://www.cs.toronto.edu/~toni/Papers/icml-final.pdf. 
    
    Adaptation of the consistency metrics implemented in: 
    https://aif360.mybluemix.net/

    Return consistency result    
    
    Parameters
    ----------      
    n_neighbors: int
        Number of neighbors to use in KNN
    """
    X = X.copy()
    X.drop(protect_attribute, axis=1, inplace=True)
    X = StandardScaler().fit_transform(X) 
    num_samples = X.shape[0]

    # learn a KNN on the features
    nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(X)

    # compute consistency score
    consistency = 0.0
    for i in range(num_samples):
        consistency += np.abs(y[i] - np.mean(y[indices[i]]))
    consistency = 1.0 - consistency/num_samples

    return consistency

def counterfactual(df, model, pa):
    df_sel = df[df[pa] == 1]
    pred = model.predict(df_sel)
    prob_y1 = pred.sum() / len(pred)
    df_inv = df_sel
    df_inv[pa] = 0
    pred_inv = model.predict(df_inv)
    prob_y1_inv =  pred_inv.sum() / len(pred_inv)
    return prob_y1_inv - prob_y1

def compute_metrics(model, X_test, y_test, X_train, y_train, dataset_test, 
                    unprivileged_groups, privileged_groups, protect_attribute, 
                    print_result):
    """
    Calculate and return: model accuracy and fairness metrics
    
    Parameters
    ----------
    model: scikit-learn classifier    
    X_test: numpy 2d array
    y_test: numpy 1d array
    X_train: numpy 2d array
    y_train: numpy 1d array
    dataset_test: aif360.datasets.BinaryLabelDataset
    unprivileged_groups: list<dict>
        Dictionary where the key is the name of the sensitive column in the 
        dataset, and the value is the value of the unprivileged group in the
        dataset
    privileged_groups: list<dict>
        Dictionary where the key is the name of the sensitive column in the 
        dataset, and the value is the value of the privileged group in the
        dataset
    protect_attribute
    print_result
    """
    result = {}
    
    y_pred_test = model.predict(X_test)
    result['acc_test'] = accuracy_score(y_true=y_test, y_pred=y_pred_test)
    y_pred_train = model.predict(X_train)
    result['acc_train'] = accuracy_score(y_true=y_train, y_pred=y_pred_train)
    
    dataset_pred = dataset_test.copy()
    dataset_pred.labels = y_pred_test

    bin_metric = BinaryLabelDatasetMetric(dataset_pred, 
                                          unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups)
    result['disp_impact'] = bin_metric.disparate_impact()
    result['stat_parity'] = bin_metric.mean_difference()

    classif_metric = ClassificationMetric(dataset_test, dataset_pred, 
                                          unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups)
    result['avg_odds'] = classif_metric.average_odds_difference()
    result['equal_opport'] = classif_metric.equal_opportunity_difference()
    result['false_discovery_rate'] = classif_metric.false_discovery_rate_difference()
    result['entropy_index'] = classif_metric.generalized_entropy_index()
    result['acc_test_clf'] = classif_metric.accuracy(privileged=None)
    result['acc_test_priv'] = classif_metric.accuracy(privileged=True)
    result['acc_test_unpriv'] = classif_metric.accuracy(privileged=False)
    
    result['consistency'] = consitency(X_test, y_pred_test, protect_attribute, n_neighbors=5)
    result['counterfactual'] = counterfactual(X_test, model, protect_attribute)
    
    if print_result:
        print("Train accuracy: ", result['acc_train'])
        print("Test accuracy: ", result['acc_test'])
        print("Test accuracy clf: ", result['acc_test_clf'])
        print("Test accuracy priv.: ", result['acc_test_priv'])
        print("Test accuracy unpriv.: ", result['acc_test_unpriv'])
        print('Disparate impact: ', result['disp_impact'])
        print('Mean difference: ', result['stat_parity'])
        print('Average odds difference:', result['avg_odds'])
        print('Equality of opportunity:', result['equal_opport'])
        print('False discovery rate difference:', result['false_discovery_rate'])
        print('Generalized entropy index:', result['entropy_index'])
        print('Consistency: ', result['consistency'])
        print('Counterfactual fairness: ', result['counterfactual'])

    return result


def fairCorrectUnder(df, pa, label, fav, d=1):
    """Correct the proportion of positive cases for favoured and unfavoured subgroups through
    subsampling the favoured positive and unfavoured negative classes. Parameter d should be
    a number between -1 and 1 for this to work properly."""
        
    # subset favoured positive, favoured negative, unfavoured positive, unfavoured negative
    fav_pos = df[(df[pa] == fav) & (df[label] == 1)]
    fav_neg = df[(df[pa] == fav) & (df[label] == 0)]
    unfav_pos = df[(df[pa] != fav) & (df[label] == 1)]
    unfav_neg = df[(df[pa] != fav) & (df[label] == 0)]
    
    # get favoured and unfavoured number of rows
    fav_size = fav_pos.shape[0] + fav_neg.shape[0]
    unfav_size = unfav_pos.shape[0] + unfav_neg.shape[0]

    # get positive ratios for favoured and unfavoured
    fav_pr = fav_pos.shape[0] / fav_size
    unfav_pr = unfav_pos.shape[0] / unfav_size
    pr = df[df[label] == 1].shape[0] / df.shape[0]

    # coefficients for fitting quad function
    a = ((fav_pr + unfav_pr) / 2) - pr
    b = (fav_pr - unfav_pr) / 2
    c = pr

    # corrected ratios
    corr_fpr = (a * (d ** 2)) + (b * d) + c
    corr_upr = (a * (d ** 2)) - (b * d) + c
    
    # correcting constants
    fav_k = corr_fpr / (1 - corr_fpr)
    unfav_k = (1 - corr_upr) / corr_upr
    
    # sample sizes for fav_pos and unfav_neg
    fav_pos_size = math.floor(fav_neg.shape[0] * fav_k)
    unfav_neg_size = math.floor(unfav_pos.shape[0] * unfav_k)
    
    # samples from fav_pos and unfav_neg to correct proportions
    corr_fav_pos = fav_pos.sample(fav_pos_size)
    corr_unfav_neg = unfav_neg.sample(unfav_neg_size)
    
    # concatenate df's
    corr_dfs = [corr_fav_pos, fav_neg, unfav_pos, corr_unfav_neg]
    corr_df = pd.concat(corr_dfs)
    
    return corr_df
