import pandas as pd
import json
from sklearn import metrics
from scipy import stats
import numpy as np

# import results and test targets DIABETES
df =  pd.read_csv('data_and_results/diabetes/diabetes_probs_hsgs.csv') ##########
df.reset_index(inplace=True)

with open('data_and_results/diabetes//test_data.json') as json_file:
    y_test = json.load(json_file)[1]


# import results and test targets NON-LINEAR dataset
df = pd.read_csv('data_and_results/non_linear/experiments_nonlinear_probs.csv')
df.reset_index(inplace=True)

with open('data_and_results/non_linear/test_nonlinear.json') as json_file:
    y_test = json.load(json_file)[1]


# import results and test targets XOR dataset
df =  pd.read_csv('data_and_results/XOR/experiments_XOR_probs.csv') 
df.reset_index(inplace=True)

with open('data_and_results/XOR/test_xor.json') as json_file:
    y_test = json.load(json_file)[1]



#=============== GET METRICS


#len(json.loads(df['neuron_outputs'][0]))
#len(y_test*5)

# get metrics
get_metrics = {'roc_auc_score':[], 'top_k_accuracy_score':[], 'KS':[], 'KS_pvalue':[]}
for i in range(len(df)):
    predicted = json.loads(df['neuron_outputs'][i])
    if len(predicted) == len(y_test*10):
        y_true = y_test*10
    elif len(predicted) == len(y_test*5):
        y_true = y_test*5
    else:
        y_true = y_test
    get_metrics['roc_auc_score'].append(metrics.roc_auc_score(y_true, predicted))
    get_metrics['top_k_accuracy_score'].append(metrics.top_k_accuracy_score(y_true, predicted, k=10))
    # get KS
    indicesa = [i for i, x in enumerate(y_true) if x == 0]
    a = list(map(predicted.__getitem__, indicesa))
    indicesb = [i for i, x in enumerate(y_true) if x == 1]
    b = list(map(predicted.__getitem__, indicesb))
    get_metrics['KS'].append(stats.ks_2samp(a, b)[0])
    get_metrics['KS_pvalue'].append(stats.ks_2samp(a, b)[1])


df = pd.concat([df, pd.DataFrame(get_metrics)], axis=1)

#df['precision_score'] = df['precision_score'].replace(1.0, 0.0)
#df['recall_score'] = df['recall_score'].replace(1.0, 0.0)


a1 = pd.DataFrame(df[['model', 'phase_strategy', 'roc_auc_score']].groupby(['model', 'phase_strategy']).max()).reset_index()
b1 = pd.DataFrame(df[['model', 'phase_strategy', 'top_k_accuracy_score']].groupby(['model', 'phase_strategy']).max()).reset_index()
c1 = pd.DataFrame(df[['model', 'phase_strategy', 'KS']].groupby(['model', 'phase_strategy']).max()).reset_index()
d1 = pd.DataFrame(df[['model', 'phase_strategy', 'KS_pvalue']].groupby(['model', 'phase_strategy']).min()).reset_index()


a1.columns = ['model', 'phase_strategy', 'best_roc_auc_score']
b1.columns = ['model', 'phase_strategy', 'best_top_k_accuracy_score']
c1.columns = ['model', 'phase_strategy', 'best_KS']
d1.columns = ['model', 'phase_strategy', 'best_KS_pvalue']

results = pd.concat([a1,  
                        b1['best_top_k_accuracy_score'], 
                        c1['best_KS'], 
                        d1['best_KS_pvalue']], 
                        axis=1)


for i in results.columns[2:]:
    results[i] = round(results[i], 2)

results.to_csv('data_and_results/table_nonlinear_metrics2.csv', index=False)


