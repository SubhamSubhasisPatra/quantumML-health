import pandas as pd
import json
from sklearn import metrics


# import results and test targets DIABETES
df = pd.concat([pd.read_csv('results/version4/results_experiment_diabetes_original.csv'), 
          pd.read_csv('results/version4/results_experiment_diabetes_original_2.csv'),
          pd.read_csv('results/version4/results_experiment_diabetes_angle2.csv'),
          pd.read_csv('results/version4/results_experiment_diabetes_angleradius.csv'),
          pd.read_csv('results/version4/results_experiment_diabetes_radius.csv')])
df.reset_index(inplace=True)
with open('results/version4/test_data.json') as json_file:
    y_test = json.load(json_file)[1]

# import results and test targets NON-LINEAR dataset
df = pd.concat([pd.read_csv('results/version6/experiment_non_linear_v2.csv'), 
          pd.read_csv('results/version6/experiment_non_linear_v2.csv'),
          pd.read_csv('results/version6/experiment_nonlinear_hsgs.csv')])
df.reset_index(inplace=True)
with open('results/version6/test_nonlinear.json') as json_file:
    y_test = json.load(json_file)[1]

# import results and test targets XOR dataset
df = pd.concat([pd.read_csv('results/version7/experiment_XOR_hsgs.csv'), 
          pd.read_csv('results/version7/experiment_XOR_phase.csv')])
df.reset_index(inplace=True)
with open('results/version7/test_xor.json') as json_file:
    y_test = json.load(json_file)[1]

# get metrics
y_true = y_test*5
get_metrics = {'precision_score':[], 'accuracy_score':[], 'recall_score':[], 'f1_score':[]}
for i in range(len(df)):
    predicted = json.loads(df['neuron_outputs'][i])
    get_metrics['precision_score'].append(metrics.precision_score(y_true, predicted))
    get_metrics['accuracy_score'].append(metrics.accuracy_score(y_true, predicted))
    get_metrics['recall_score'].append(metrics.recall_score(y_true, predicted))
    get_metrics['f1_score'].append(metrics.f1_score(y_true, predicted))


df = pd.concat([df, pd.DataFrame(get_metrics)], axis=1)

a1 = pd.DataFrame(df[['model', 'phase_strategy', 'accuracy_score']].groupby(['model', 'phase_strategy']).mean()).reset_index()
a2 = pd.DataFrame(df[['model', 'phase_strategy', 'accuracy_score']].groupby(['model', 'phase_strategy']).max()).reset_index()
a3 = pd.DataFrame(df[['model', 'phase_strategy', 'accuracy_score']].groupby(['model', 'phase_strategy']).min()).reset_index()
a4 = pd.DataFrame(df[['model', 'phase_strategy', 'accuracy_score']].groupby(['model', 'phase_strategy']).var()).reset_index()
b1 = pd.DataFrame(df[['model', 'phase_strategy', 'precision_score']].groupby(['model', 'phase_strategy']).mean()).reset_index()
c1 = pd.DataFrame(df[['model', 'phase_strategy', 'recall_score']].groupby(['model', 'phase_strategy']).mean()).reset_index()
d1 = pd.DataFrame(df[['model', 'phase_strategy', 'f1_score']].groupby(['model', 'phase_strategy']).mean()).reset_index()


a1.columns = ['model', 'phase_strategy', 'avg_accuracy_score']
a2.columns = ['model', 'phase_strategy', 'max_accuracy_score']
a3.columns = ['model', 'phase_strategy', 'min_accuracy_score']
a4.columns = ['model', 'phase_strategy', 'var_accuracy_score']
b1.columns = ['model', 'phase_strategy', 'avg_precision_score']
c1.columns = ['model', 'phase_strategy', 'avg_recall_score']
d1.columns = ['model', 'phase_strategy', 'avg_f1_score']

results = pd.concat([a1, a2['max_accuracy_score'], 
                        a3['min_accuracy_score'], 
                        a4['var_accuracy_score'], 
                        b1['avg_precision_score'], 
                        c1['avg_recall_score'], 
                        d1['avg_f1_score']], 
                        axis=1)

results = results.drop([0, 1, 3])

for i in results.columns[2:]:
    results[i] = round(results[i], 2)

results.to_csv('results/table_XOR.csv', index=False)


df[df.phase_strategy == 'angleradius']