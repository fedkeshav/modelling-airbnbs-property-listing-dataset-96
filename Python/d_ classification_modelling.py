#%%
from b_modelling_functions import load_and_standardise_numerical_data, BestChoiceModels
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# Encode the labels
def label_encoding(y_train: pd.Series, y_test: pd.Series) -> tuple:
    '''
    Encodes the labels into numbers

    Inputs
    ---------------
    y_train: Training data for labels
    y_test: Test data for labels

    Returns
    ---------------
    y_train_encoded: Encoded labels for training data
    y_test_encoded: Encoded labels for test data
    '''
    le = LabelEncoder()
    le.fit(y_train)
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    return y_train_encoded, y_test_encoded


#%%
from sklearn.model_selection import GridSearchCV
X_train_scaled, X_test_scaled, y_train, y_test = load_and_standardise_numerical_data('Category')
y_train_encoded, y_test_encoded = label_encoding(y_train, y_test)
model_instance = DecisionTreeClassifier()
params_dict = {
    'random_state': [42]
    }
class_scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
class_refit_metric = 'accuracy'
grid_search = GridSearchCV(model_instance, param_grid = params_dict, scoring = class_scoring_metrics, refit = class_refit_metric, cv = 3)
grid_search.fit(X_train_scaled, y_train_encoded)
best_model = grid_search.best_estimator_
best_hyperparameter_dict = grid_search.best_params_
best_metrics = grid_search.best_score_
print(best_hyperparameter_dict)
print(best_metrics)
print(y_train_encoded[0:4])
#%%
# Define all the model classes and their hyperparameters to test    

classification_model_classes = [LogisticRegression(),DecisionTreeClassifier(), 
                                RandomForestClassifier(), GradientBoostingClassifier()]

logreg_hyperparams_dict = {
    'penalty': ['l1', 'l2'],
    'random_state': [42]
}

decision_tree_hyperparams_dict = {
    'min_samples_split': [5, 10, 20],
    'random_state': [42]
    }

rf_hyperparams_dict = {
    'n_estimators': [100, 200],
    'random_state': [42]
    }

gb_hyperparams_dict = {
    'n_estimators': [100, 200],
    'random_state': [42]
    }

hyperparameters = [logreg_hyperparams_dict, decision_tree_hyperparams_dict,
                   rf_hyperparams_dict, gb_hyperparams_dict]
classification_model_class_and_hyperparameters = dict(zip(classification_model_classes, hyperparameters))

# Defining scoring and refit metrics
model_type = 'classification'
class_scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
class_refit_metric = 'accuracy'

#%%
if __name__ == '__main__':
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_standardise_numerical_data('Category')
    y_train_encoded, y_test_encoded = label_encoding(y_train, y_test)
    bcm = BestChoiceModels()
    bcm.evaluate_all_models(model_type = model_type,
                        model_class_and_hyperparameters= classification_model_class_and_hyperparameters,
                        X_train = X_train_scaled, y_train = y_train_encoded,
                        scoring_metrics = class_scoring_metrics,
                        refit_metric = class_refit_metric,
                        folds = 3)
#%%
    final_model, final_hyperparams, final_metrics = bcm.find_best_model(model_type = model_type, 
                                                                    model_class_and_hyperparameters = classification_model_class_and_hyperparameters, 
                                                                    scoring_metrics = class_scoring_metrics,
                                                                    optimisation_metric = class_refit_metric)
    print(f' Best model is {final_model}')
    print(f' Best hyperparameters are {final_hyperparams}')
    print(f' Best metrics are {final_metrics}')

#%%
    
logreg_hyperparams_dict = {
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000, 2000],
    'multi_class': ['ovr'],
    'random_state': [42]
}

decision_tree_hyperparams_dict = {
    'min_samples_split': [5, 10, 20],
    'ccp_alpha': [100, 1000, 10000],
    'random_state': [42]
    }

rf_hyperparams_dict = {
    'n_estimators': [100, 200],
    'min_samples_split': [10, 20],
    'max_features': [2 , 3],
    'ccp_alpha': [100, 1000],
    'random_state': [42]
    }

gb_hyperparams_dict = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.2],
    'subsample': [0.5, 1],
    'min_samples_split': [10, 20],
    'ccp_alpha': [100, 1000],
    'random_state': [42]
    }