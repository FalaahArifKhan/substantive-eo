import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utils.common_helpers import set_protected_groups, compute_prevalence


class GenericCompetition():
    def __init__(self, applicant_pool, sensitive_attribute_names, priv_values):
        self.applicant_pool = applicant_pool
        self.sensitive_attribute_names = sensitive_attribute_names
        self.priv_values = priv_values
        self.social_groups = None
        #self.social_groups = set_protected_groups(applicant_pool, sensitive_attribute_names, priv_values)
        self.selected = None
        self.overall_prevalence = None
        self.contest_chances_disparity = None
        self.metrics = None
        self.target_name = None

    def set_selected_pool(self, selected):
        self.selected = selected
        self.social_groups = set_protected_groups(self.selected, self.sensitive_attribute_names, self.priv_values)
        return True

    def compute_metrics(self, selected, target_name, pos_value, refit=True):
        self.target_name = target_name
        self.metrics = pd.DataFrame({"metric_name":[], "group_name":[], "metric_values":[]})
        if (self.selected is None) or (refit is True):
            self.selected = selected
        
        if self.social_groups is None or refit is True:
            self.social_groups = set_protected_groups(selected, self.sensitive_attribute_names, self.priv_values)

        df_prevalences = []
        self.overall_prevalence = compute_prevalence(self.selected, target_name, pos_value)
        
        df_prevalences = []
        for group_name in  self.social_groups.keys():
            temp = self.social_groups[group_name]
            df_prevalences.append(compute_prevalence(temp, target_name, pos_value))

        df_selection_rate = np.array(df_prevalences)/self.overall_prevalence
        
        self.metrics["metric_name"] = ["prevalence"]*len(self.social_groups.keys()) + ["selection_rate"]*len(self.social_groups.keys())
        self.metrics["group_name"] = list(self.social_groups.keys())*2
        self.metrics["metric_values"] = list(df_prevalences) + list(df_selection_rate)

        return self.metrics

    def compute_contest_chances_disparity(self, groups_info):
        if self.metrics is None:
            raise Exception("Compute metrics first")

        res = {}
        for protected_group_name in groups_info.keys():
            group_names = groups_info[protected_group_name]
            temp = [self.metrics[(self.metrics['metric_name']=='selection_rate') & (self.metrics['group_name']==x)].metric_values.values[0] for x in group_names]
            res[protected_group_name] = np.max(temp) - np.min(temp)

        self.contest_chances_disparity = res
        return self.contest_chances_disparity

'''
For data generated from the initialize_BN() function:
	SES: 0 - high, 1- med, 2 - low
	SEX: 0 - priv, 1 - dis
	SCHOOL: 0 - high, 1 - med, 2 - low
	SAT: 0 - high, 1 - med, 2 - low
	CGPA: 0 - high, 1 - med, 2 - low
    COLLEGE: 0 - yes, 1 - no
	INTERN: 0 - yes, 1 - no
	JOB: 0 - yes, 1 - no

    Protected groups: Priv: (SEX,0), (SES,0)
'''
class Competition():
    def __init__(self, dataset, covariates, target, protected_groups, priv_values):
        #self.protected_groups = protected_groups
        #self.priv_values = priv_values
        self.features = covariates
        self.target = target
        self.X_data = dataset[covariates]
        self.y_data = dataset[target]
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.protected_groups = set_protected_groups(dataset, protected_groups, priv_values)
        self.metrics = None
        self.results = None
        self.base_model = None

    def create_train_test_val_split(self, SEED):
        X_, X_test, y_, y_test = train_test_split(self.X_data, self.y_data, test_size=0.2, random_state=SEED)
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.25, random_state=SEED)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        #self.test_groups = set_protected_groups(self.X_test, self.protected_groups, self.priv_values)
        self.results = X_test.copy(deep=True)
        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val

    def set_base_model(self, model):
        self.base_model = model 

    def fit_base_model(self, base_model):
        if base_model is None:
            if self.base_model is None:
                raise Exception("Set a base model or pass one explicitly")
                return
        else:
            self.base_model = base_model

        self.base_model.fit(self.X_train, self.y_train)
        #self.results['base'] = self.base_model.predict(self.X_test)
        #return self.results
        return self.base_model.score(self.X_test, self.y_test)

    def predict(self, test_sample):
        return self.base_model.predict(test_sample)

    def predict_proba(self, test_sample):
        return self.base_model.predict_proba(test_sample)





  