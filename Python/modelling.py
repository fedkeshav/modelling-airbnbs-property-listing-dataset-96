#%%
from dotenv import load_dotenv
import itertools
import joblib
import json
import numpy as np
import os
import pandas as pd
from tabular_data import load_airbnb_data
from scipy.stats import randint
from sklearn.linear_model import SGDRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import typing


class CustomHyperparameterTuning():
    '''
    A class containing all the methods from scratch to tune hyperparameters
    '''
    def __init__():
        pass

    def grid_search(self, hyperparameter_dict: typing.Dict[str, list]):
        keys, values = zip(*hyperparameter_dict.items())
        yield from (dict(zip(keys, v)) for v in itertools.product(*values))

    def k_fold(self, datasets, n_splits: int = 5):
        chunks = np.array_split(datasets, n_splits)
        for i in range(n_splits):
            training = chunks[:i] + chunks[i + 1:]
            validation = chunks[i]
            yield np.concatenate(training), validation

    def custom_tune_regression_model_hyperparameters(self, model_class: str, x_train: pd.DataFrame, y_train: pd.Series, hyperparameter_dict: dict, folds: int = 5): 
        '''
        Custom model that finds the best hyperparameters for a model using grid search and cross-validation

        Inputs
        --------------
        model_class: The chosen model class (e.g., SGDRegressor())
        
        x_train: Training data
        
        y_train: Labels
        
        hyperparameter_dict: Dictionary containing different values to test for hyperparameters
        
        Returns
        --------------
        best_model: Model instantitated with best hyperparameter values
        
        best_hyperparameters_dict: A dictionary containing optimal hyperparameters

        best_metrics_dict: A dictionary containing the best validation metrics
        '''
        best_model, best_hyperparams = None, None  ## DO WE NEED THIS????
        best_validation_rsquare, best_validation_mse = np.inf, np.inf
        best_metrics_dict = {'validation_MSE': best_validation_mse, 'validation_rsquare': best_validation_rsquare} 

        # First loop over each of the hyperparameter combinations
        for hyperparams in self.grid_search(hyperparameter_dict):
            validation_mse = 0
            validation_rsquare = 0

            # Now loop over each of the 5 splits of data for cross-validation
            for (X_training_scaled, X_validation_scaled), (y_training, y_validation) in zip(
                self.k_fold(X_train_scaled, folds), self.k_fold(y_train, folds)):    
                model = model_class(**hyperparams, random_state = 42)
                model.fit(X_training_scaled, y_training)
                y_validation_pred = model.predict(X_validation_scaled)
                fold_mse = mean_squared_error(y_validation, y_validation_pred)
                fold_rsquare = r2_score(y_validation, y_validation_pred)
                validation_mse += fold_mse
                validation_rsquare += fold_rsquare
            # Average the error across the splits and find if the average is lowest for this hyperparameter combination
            average_validation_mse = validation_mse / n_splits
            average_validation_rsquare = validation_rsquare / n_splits
            if average_validation_mse < best_validation_mse:
                best_metrics_dict['validation_MSE'] = average_validation_mse
                best_metrics_dict['validation_rsquare'] = average_validation_rsquare
                best_hyperparams = hyperparams
                best_model = model_class(**best_hyperparams, random_state = 42)
        return best_model, best_hyperparams, best_metrics_dict

def tune_regression_model_hyperparameters(model_class: str, X_train: pd.DataFrame, y_train: pd.Series, params_dict: dict, folds: int = 5) -> tuple:
    '''
    Finds the best hyperparameters for a model using sklearn's grid search and cross-validation

    Inputs
    --------------
    model_class: The chosen model class (e.g., SGDRegressor())
    
    x_train: Training data
    
    y_train: Labels
    
    params_grid: Dictionary containing grid of values to test for hyperparameters
    
    folds: number of folds for cross-validation

    Returns
    --------------
    best_model: Model instantitated with best hyperparameter values
    
    best_hyperparams: A dictionary containing optimal hyperparameters

    best_metrics_dict: A dictionary containing the best validation metrics
    '''
    model = model_class
    scoring_metrics = ['neg_mean_squared_error', 'r2']
    grid_search = GridSearchCV(model, param_grid = params_dict, cv = folds, scoring = scoring_metrics, refit='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_hyperparameter_dict = grid_search.best_params_
    best_metrics_dict = grid_search.best_score_
    return best_model, best_hyperparameter_dict, best_metrics_dict

def save_model(model_type : str , model_class: str,  best_model, best_hyperparameter: dict, best_metrics: dict) -> None: 
    '''
    Save model, hyperparameters and metrics after tuning

    Inputs:
    --------------
    model_type: can be either 'regression' or 'classification' [KP: Automate errors and show possible values in inputs]
    
    model_class: can be any of the model classes. E.g., SGDRegressor()
    
    best_model: model that performs best along with its hyperparameters
    
    best_hyperparameter: best performing hyperparameter according to evaluation metric
    
    best_metrics: best validation metrics
    
    Returns:
    --------------
    Nothing
    
    '''
    load_dotenv() # Load environment variables from .env file
    models_folder = os.getenv(f'MODELS_DIR') # Imports directory path from .env file
    models_subfolder = os.path.join(models_folder, model_type, model_class)
    if os.path.exists(models_subfolder) == False:
        os.mkdir(models_subfolder)

    # Save the model, hyperparameters and metrics in the above folder
    joblib.dump(best_model, os.path.join(models_subfolder, 'model.joblib'))  
    with open(os.path.join(models_subfolder, 'hyperparameters.json'), 'w') as json_file: 
        json.dump(best_hyperparameter, json_file)
    with open(os.path.join(models_subfolder, 'metrics.json'), 'w') as json_file: 
        json.dump(best_metrics, json_file)

def evaluate_all_models(model_class_and_hyperparameters: dict, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    '''
    Chooses optimal hyperparameters of each model and saves the best version of each model
    
    Inputs: 
    --------------
    model_class_and_hyperparameters: A dictionary of all model classes as keys and their associated values as dictionary of hyperparameter values for tuning
    
    X_train: training dataset
    
    y_train: labels for training
            
    Returns:
    --------------
    Nothing
    '''
    model_classes = model_class_and_hyperparameters.keys()

    # For each model class, tune the model with dictionary of hyperparameters
    for model_class in model_classes:
        print(f'Evaluating model class of {model_class}')
        print('----------------------------------------')
        hyperparameters_dict = model_class_and_hyperparameters[model_class]
        best_model, best_hyperparameters, best_metrics = tune_regression_model_hyperparameters(model_class, 
                                                                                               X_train, y_train, 
                                                                            params_dict = hyperparameters_dict)
        # Save the best version of the model
        save_model(model_type = 'regression', model_class = f'{model_class}', 
                   best_model = best_model, best_hyperparameter = best_hyperparameters, best_metrics = best_metrics)

def find_best_model(model_type: str, model_class_and_hyperparameters: dict):
    '''
    Finds the best model out of all evaluated models
    
    Inputs: 
    --------------
    model_type: this can take either 'regression' or 'classification'
    model_class_and_hyperparameters: A dictionary of all model classes and their associated dictionary of hyperparameter values for tuning           
    
    Returns:
    --------------
    Best model
    Associated dictionary of hyperparameters
    Dictionary of performance metrics
    '''
    best_model = None
    best_model_hyperparamets = None
    best_model_mse = float('inf')
    load_dotenv() # Load environment variables from .env file
    models_folder = os.getenv(f'MODELS_DIR') # Imports directory path from .env file
    models_subfolder = os.path.join(models_folder, model_type)

    for model_class in model_class_and_hyperparameters.keys():
        folder = os.path.join(models_subfolder, f'{model_class}')
        metrics_filepath = os.path.join(folder, 'metrics.json')
        # Load the metrics from the file
        with open(metrics_filepath, 'r') as file:
            all_metrics = json.load(file)
            mean_squared_error = abs(all_metrics[0])
        #mean_square_error = metrics_data.get('mean_square_error', float('inf'))
        print(f'{model_class}: {mean_squared_error}')
        if mean_squared_error < best_model_mse:
            best_model_mse = mean_squared_error
            best_model = model_class
        
    best_model_folder = os.path.join(models_subfolder, f'{best_model}')
    model_path = os.path.join(best_model_folder, 'model.joblib')
    metrics_path = os.path.join(best_model_folder, 'metrics.json')
    hyperparameters_path = os.path.join(best_model_folder, 'hyperparameters.json')

    model = joblib.load(model_path)
    with open(metrics_path, 'r') as metrics_file:
        metrics = json.load(metrics_file)
    with open(hyperparameters_path, 'r') as hyperparams_file:
        hyperparameters = json.load(hyperparams_file)

    return model, hyperparameters, metrics


# Load, split and standardise data
X, y = load_airbnb_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)        

# Define all the model classes and their hyperparameters to test    
model_classes = [SGDRegressor(),DecisionTreeRegressor(), 
                 RandomForestRegressor(),
                 AdaBoostRegressor(), GradientBoostingRegressor()]

sgd_hyperparams_dict = {
    'loss': ['squared_error'],
    'alpha': [0.05,  0.1] ,
    'learning_rate': ['constant', 'optimal', 'invscaling'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'random_state': [42]    
    }

decision_tree_hyperparams_dict = {
    'criterion': ['squared_error'],
    'min_samples_split': [5, 10, 20],
    'ccp_alpha': [100, 1000, 10000],
    'random_state': [42]
}

rf_hyperparams_dict = {
    'criterion': ['squared_error'],
    'n_estimators': [100, 200],
    'min_samples_split': [10, 20],
    'max_features': [2 , 3],
    'ccp_alpha': [100, 1000],
    'random_state': [42]
}

adaboost_hyperparams_dict = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.5, 1, 2],
    'random_state': [42]
  }

gb_hyperparams_dict = {
    'n_estimators': [50,100, 150, 200],
    'learning_rate': [0.05, 0.1, 0.5],
    'random_state': [42]
}

hyperparameters = [sgd_hyperparams_dict, decision_tree_hyperparams_dict,
                   rf_hyperparams_dict, adaboost_hyperparams_dict, gb_hyperparams_dict]

model_class_and_hyperparameters = dict(zip(model_classes, hyperparameters))

#%%
# Evaluate the best version of all models
if __name__ == "__main__":
    evaluate_all_models(model_class_and_hyperparameters, X_train_scaled, y_train)

#%%
final_model, final_hyperparams, final_metrics = find_best_model(model_type = 'regression', model_class_and_hyperparameters = model_class_and_hyperparameters)

'''
1. General cleaning and organisatiton
2. Restrict input values to regression and classification
3. Return error if weird input values provided
4. Read Gradient Boost parameters
5. Solve doubts
6. Best metric is an ndarray - need to make it to dictionary for things to work
'''
#%%

# Load, split and standardise data
X, y = load_airbnb_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)        

# Define all the model classes and their hyperparameters to test    
model_class = SGDRegressor()

sgd_hyperparams_dict = {
    'loss': ['squared_error'],
    'alpha': [0.05,  0.1] ,
    'learning_rate': ['constant', 'optimal', 'invscaling'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'random_state': [42]    
    }
final_model, final_hyperparams, final_metrics = tune_regression_model_hyperparameters(model_class, X_train_scaled, y_train, sgd_hyperparams_dict)
print(final_metrics)
print(type(final_metrics))
# %%
print(final_metrics)
print(type(final_metrics))
# %%
