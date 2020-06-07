import glob
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as pl
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def find_path(lst_path, name):
    return [var for var in lst_path if name in var]

class RunSHAP():
    
    def __init__(self, model, X_train, X_test, protected_attribute, 
                 name):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.protected_attribute = protected_attribute
        self.name = name
        
    def _run(self, shap_values, result, print_result):
        pl.figure()
        shap.summary_plot(shap_values, self.X_test, 
                          feature_names=self.X_test.columns,
                          plot_type='dot', show=False)
        pl.savefig('result_image/summaryplot_{}_{}.png'.format(result['shap_method'], 
                                                               self.name))

        df_feat = pd.DataFrame({'feature_names': self.X_test.columns, 
                                'value': np.abs(shap_values).mean(axis=0)})
        df_feat = df_feat.sort_values(by='value', ascending=False).reset_index(drop=True)
        result['feat_import'] = df_feat.loc[df_feat['feature_names'] == 
                                            self.protected_attribute, 'value'].iloc[0]
        result['feature_pos'] = df_feat[df_feat['feature_names'] == 
                                        self.protected_attribute].index[0]

        pl.figure()
        shap.dependence_plot(self.protected_attribute, shap_values, self.X_test, 
                             self.X_test.columns, show=False)
        pl.savefig('result_image/dependenceplot_{}_{}.png'.format(result['shap_method'], 
                                                                  self.name))

        df = pd.DataFrame({'shap': shap_values[:, 0], 
                           'feat_value': self.X_test[self.protected_attribute]})
        result['unpriv_value'] = df.loc[df['feat_value'] == 0, 'shap'].mean()
        result['priv_value'] = df.loc[df['feat_value'] == 1, 'shap'].mean()

        if print_result:
            print('{} SHAP results for {}:'.format(result['shap_method'], self.name))
            print('Feature importance: ', result['feat_import'])
            print('Feature position: ', result['feature_pos'])
            print('Mean SHAP value unprivileged class: ', result['unpriv_value'])
            print('Mean SHAP value privileged class: ', result['priv_value'])

        return result        
        
    def tree(self, print_result=False):
        result = {}
        result['name'] = self.name
        result['shap_method'] = 'tree'
        
        explainer = shap.TreeExplainer(model=self.model, data=self.X_train, 
                                       feature_dependence='tree_path_dependent')
        shap_values = explainer.shap_values(self.X_test)
        if isinstance(shap_values,list):
            shap_values = shap_values[1]
        
        return self._run(shap_values, result, print_result)

    def linear(self, print_result=False):
        result = {}
        result['name'] = self.name
        result['shap_method'] = 'linear'

        explainer = shap.LinearExplainer(self.model, self.X_train, 
                                         feature_dependence="correlation")
        shap_values = explainer.shap_values(self.X_test)
        
        return self._run(shap_values, result, print_result)
    
    def kernel(self, print_result=False):
        result = {}
        result['name'] = self.name
        result['shap_method'] = 'kernel'

        X_train_summary = shap.kmeans(self.X_train, 50)
        explainer = shap.KernelExplainer(model=self.model.predict_proba, data=X_train_summary)
        shap_values = explainer.shap_values(self.X_test)
        if isinstance(shap_values,list):
            shap_values = shap_values[1]
        
        return self._run(shap_values, result, print_result)    

def iterate_shap_methods(models_path_sel, X_train, X_test, protected_attribute):
    result_full = []
    for model_path in models_path_sel:
        result = []
        print(model_path)
        model = pickle.load(open(model_path, 'rb'))
        name = model_path.split('.')[0].split('/')[1]

        shap_run = RunSHAP(model, X_train, X_test, protected_attribute, name)
        # Tree SHAP
        three_methods = ['gb', 'rf']
        if name.split('_')[-2] in three_methods:
            res = shap_run.tree(print_result=False)
            result.append(res)

        # Linear SHAP
        linear_methods = ['lr']
        if name.split('_')[-2] in linear_methods:
            res = shap_run.linear(print_result=False)
            result.append(res)

        # Kernel SHAP
        res = shap_run.kernel(print_result=False)
        result.append(res)

        pickle.dump(result, open('res_temp/' + model_path.split('/')[1], "wb"))
        result_full += result
    return result_full        

def run_process(dataset_name, protected_attribute, label):
    # models_path = glob.glob('models/{}*.pkl'.format(dataset_name))
    models_path = glob.glob('models/{}*.pkl'.format(dataset_name))
    res_salvos = glob.glob('res_temp/{}*.pkl'.format(dataset_name))
    result = []
    if res_salvos:
        paths_salvos = ['models/{}'.format(res.split('/')[1]) for res in res_salvos]
        models_path = list(set(models_path) - set(paths_salvos))
        for res in res_salvos:
            result += pickle.load(open(res, 'rb'))

    print("Total paths executar: ", len(models_path))

    df_train = pd.read_csv('data/{}_train.csv'.format(dataset_name))
    df_test = pd.read_csv('data/{}_test.csv'.format(dataset_name))
    df_train_usdmin1 = pd.read_csv('data/{}_train_usd-1.csv'.format(dataset_name))
    df_train_usd0 = pd.read_csv('data/{}_train_usd0.csv'.format(dataset_name))

#     result = []

    X_train = df_train.drop(label, axis=1)
    X_test = df_test.drop(label, axis=1)

    # Model with bias
    models_path_sel = find_path(models_path, 'orig')
    result += iterate_shap_methods(models_path_sel, X_train, X_test, 
                                            protected_attribute)

    # Model with reweghing
    models_path_sel = find_path(models_path, 'rw')
    result += iterate_shap_methods(models_path_sel, X_train, X_test, 
                                            protected_attribute)

    # Model with undersampling with d=-1
    X_train = df_train_usdmin1.drop(label, axis=1)
    models_path_sel = find_path(models_path, 'usd-1')
    result += iterate_shap_methods(models_path_sel, X_train, X_test, 
                                            protected_attribute)

    # Model with undersampling with d=0
    X_train = df_train_usd0.drop(label, axis=1)
    models_path_sel = find_path(models_path, 'usd0')
    result += iterate_shap_methods(models_path_sel, X_train, X_test, 
                                            protected_attribute)

    # save result
    df_result = pd.DataFrame(result)
    df_result.to_csv('data/result_shap_{}.csv'.format(dataset_name), index=False)

    return df_result
