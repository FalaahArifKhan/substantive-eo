import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix

def partition_by_group_intersectional(df, column_names, priv_values):
    priv_priv = df[(df[column_names[0]] == priv_values[0]) & (df[column_names[1]] == priv_values[1])]
    priv_dis = df[(df[column_names[0]] == priv_values[0]) & (df[column_names[1]] != priv_values[1])]
    dis_priv = df[(df[column_names[0]] != priv_values[0]) & (df[column_names[1]] == priv_values[1])]
    dis_dis = df[(df[column_names[0]] != priv_values[0]) & (df[column_names[1]] != priv_values[1])]
    return priv_priv, priv_dis, dis_priv, dis_dis


def partition_by_group_binary(df, column_name, priv_value):
    priv = df[df[column_name] == priv_value]
    dis = df[df[column_name] != priv_value]
    if len(priv)+len(dis) != len(df):
        raise ValueError("Error! Not a partition")
    return priv, dis


def set_protected_groups(X_test, column_names, priv_values):
    groups={}
    #groups[column_names[0]+'_'+column_names[1]+'_priv'], groups[column_names[0]+'_'+column_names[1]+'_dis'] = partition_by_group_intersectional(X_test, column_names, priv_values)
    groups[column_names[0]+'_priv_'+column_names[1]+'_priv'], groups[column_names[0]+'_priv_'+column_names[1]+'_dis'], groups[column_names[0]+'_dis_'+column_names[1]+'_priv'], groups[column_names[0]+'_dis_'+column_names[1]+'_dis'] = partition_by_group_intersectional(X_test, column_names, priv_values)
    groups[column_names[0]+'_priv'], groups[column_names[0]+'_dis'] = partition_by_group_binary(X_test, column_names[0], priv_values[0])
    groups[column_names[1]+'_priv'], groups[column_names[1]+'_dis'] = partition_by_group_binary(X_test, column_names[1], priv_values[1])
    return groups

def confusion_matrix_metrics(y_true, y_preds):
    metrics={}
    TN, FP, FN, TP = confusion_matrix(y_true, y_preds, labels=[1,0]).ravel()
    metrics['TPR'] = TP/(TP+FN)
    metrics['TNR'] = TN/(TN+FP)
    metrics['PPV'] = TP/(TP+FP)
    metrics['FNR'] = FN/(FN+TP)
    metrics['FPR'] = FP/(FP+TN)
    metrics['Accuracy'] = (TP+TN)/(TP+TN+FP+FN)
    metrics['F1'] = (2*TP)/(2*TP+FP+FN)
    metrics['Selection-Rate'] = (TP+FP)/(TP+FP+TN+FN)
    metrics['Positive-Rate'] = (TP+FP)/(TP+FN) 
    return metrics

def compute_prevalence(df, target_name, pos_value=1):
    return df[df[target_name] == pos_value].shape[0]/len(df)


def score_based_set_selection(df, var_names, coeffs, k, target_name, input_columns_rank_order=None):
    '''
    df: dataframe containing the data
    var_names: list of column names to be used for scoring
    coeffs: list of coefficients/weights to be used for scoring
    k: number of candidates to be selected
    target_name: name of the column to be used for storing the selection result
    rank_order_for_selection: list of column names in the order of their importance for selection
    '''
    temp = df[var_names].dot(coeffs)/np.linalg.norm(coeffs)
    # Min-max normalization of scores
    df['selected'] =(temp-temp.min())/(temp.max()-temp.min())
    
    df[target_name] = 0

    rank_order_for_selection = ['selected']
    if input_columns_rank_order is not None:
        rank_order_for_selection+=input_columns_rank_order
    selected = df.sort_values(by=rank_order_for_selection, ascending=False)[:k]

    df.loc[selected.index, target_name] = 1
    df.drop(columns=['selected'], inplace=True)

    return df

def add_external_applicants(base_population, external_frac, base_model, SEED):
    n_external = int((len(base_population) * external_frac)/(1-external_frac))
    external_applicants = base_model.simulate(n_samples=n_external, seed=SEED)
    cols_to_add = list(set(base_population.columns) - set(external_applicants.columns))
    for col in cols_to_add:
        external_applicants[col] = -1
    
    res = base_population.append(external_applicants, ignore_index=True)
    return res

def add_qualification_from_desirable_position(df, competition_label, desirable_label, generated_score_name):
    pos_idx = df[df[competition_label] == desirable_label].index
    neg_idx = df[df[competition_label] != desirable_label].index

    S_pos = np.random.beta(a=2, b=2, size=len(pos_idx))
    S_neg = np.random.beta(a=2, b=5, size=len(neg_idx))
    df[generated_score_name] = 0

    df.loc[pos_idx, generated_score_name] = S_pos
    df.loc[neg_idx, generated_score_name] = S_neg

    # Normalizing the score
    df[generated_score_name] = (df[generated_score_name] - df[generated_score_name].min())/(df[generated_score_name].max() - df[generated_score_name].min())

    return df
