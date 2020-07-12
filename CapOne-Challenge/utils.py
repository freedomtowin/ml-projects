import os 
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd
import itertools
import sqlite3 as sq


def create_database(name):
    sql_data = "working.sqlite" #- Creates DB names SQLite
    conn = sq.connect(sql_data)
    return conn

def read_first_line(fname):
    """
    Read fist line of file
    :param filename:
    :return:
    """
    data =[]
    with open(fname, "r") as data_file:
        line = data_file.readline()
        data = line.strip().split()
    return data  

def get_file_size(fname):

    statinfo = os.stat(fname)
    size = int(statinfo.st_size)
    return size


def pandas_isnull_count(col):
    null_count = col.isnull().sum()
    return null_count


def check_potentiall_nulls(col):
    """
    data = pd.DataFrame()
    data['rand']=np.concatenate([np.random.normal(5,20,(100,)),np.ones(99)])
    data['rand'].value_counts()
    potential_null_col_names = find_potential_nulls(data['rand'])
    """
    
    if null_count==0:
        #if the number of unique values is >30 and a category represents more than 20% of the data
        #this could be an indication of a missing value

        msk = len(uniq_vals)>30 and uniq_count/col.shape[0]>0.20
        
        return uniq_vals[msk]
    
def pandas_agg_cnt_unqs(x):
    return len(np.unique(x))

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])



def classification_metrics(ypred,ytrue):
    tp = np.sum((ypred==1)&(ytrue.flatten()==1))
    fp = np.sum((ypred==1)&(ytrue.flatten()==0))
    tn = np.sum((ypred==0)&(ytrue.flatten()==0))
    fn = np.sum((ypred==0)&(ytrue.flatten()==1))
    acc = (tn+tp)/(tp+tn+fp+fn)
    return tp,fp,tn,fn,acc

def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
    
def plot_cramers_v_heatmap(df, cols):
    corrM = np.zeros((len(cols),len(cols)))

    for col1, col2 in itertools.combinations(cols, 2):
        idx1, idx2 = cols.index(col1), cols.index(col2)
        corrM[idx1, idx2] = cramers_v(pd.crosstab(df[col1].astype(np.str), df[col2].astype(np.str)).as_matrix())
        corrM[idx2, idx1] = corrM[idx1, idx2]

    corr = pd.DataFrame(corrM, index=cols, columns=cols)
    ax = sns.heatmap(corr, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
    
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
def plot_pearson_corr_heatmap(df,cols):
    
    ax = sns.heatmap(df[cols].corr().round(2), annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 1, top - 1)