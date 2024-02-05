#%%
from dotenv import load_dotenv
import itertools
import joblib
import json
import numpy as np
import os
import pandas as pd
from a_tabular_data import load_airbnb_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import typing
#%%

class CustomHyperparameterTuning():
    '''
    A class containing all the methods from scratch to tune hyperparameters
    '''

    def grid_search(self, hyperparameter_dict: typing.Dict[str, list]):
        '''
        Generates a list of all hyperparameter combinations to try out
        
        Inputs
        --------------
        
        hyperparameter_dict: Dictionary containing different values to test for hyperparameters
        
        Returns
        --------------
        Different hyperparamater combinations
        
        '''
        keys, values = zip(*hyperparameter_dict.items())
        yield from (dict(zip(keys, v)) for v in itertools.product(*values))

    def k_fold(self, datasets, n_splits: int = 5):

        '''
        Generates different datasets for cross-validtion based on number of desired splits
        
        Inputs
        --------------
        
        datasets: Data that needs to be split

        n_splits: Number of splits desired of the data (default = 5)
        
        Returns
        --------------
        Different datasets for training and validation
        
        '''
        chunks = np.array_split(datasets, n_splits)
        for i in range(n_splits):
            training = chunks[:i] + chunks[i + 1:]
            validation = chunks[i]
            yield np.concatenate(training), validation

    def custom_tune_regression_model_hyperparameters(self, model_class: str, X_train: pd.DataFrame, y_train: pd.Series, hyperparameter_dict: dict, folds: int = 5): 
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

            # Now loop over each of the splits of data for cross-validation
            for (X_training_scaled, X_validation_scaled), (y_training, y_validation) in zip(
                self.k_fold(X_train, folds), self.k_fold(y_train, folds)):    
                model = model_class(**hyperparams, random_state = 42)
                model.fit(X_training_scaled, y_training)
                y_validation_pred = model.predict(X_validation_scaled)
                fold_mse = mean_squared_error(y_validation, y_validation_pred)
                fold_rsquare = r2_score(y_validation, y_validation_pred)
                validation_mse += fold_mse
                validation_rsquare += fold_rsquare
            # Average the error across the splits and find if the average is lowest for this hyperparameter combination
            average_validation_mse = validation_mse / folds
            average_validation_rsquare = validation_rsquare / folds
            if average_validation_mse < best_validation_mse:
                best_metrics_dict['validation_MSE'] = average_validation_mse
                best_metrics_dict['validation_rsquare'] = average_validation_rsquare
                best_hyperparams = hyperparams
                best_model = model_class(**best_hyperparams, random_state = 42)
        return best_model, best_hyperparams, best_metrics_dict


class BestChoiceModels():
    '''
    A class containing all the methods to tune hyperparameters and find the best model and hyperparameters associated
    '''

    def __init__(self):
        pass

    def tune_model_hyperparameters(self, model_class: str, X_train: pd.DataFrame, y_train: pd.Series, params_dict: dict, 
                                scoring_metrics: list, refit_metric: str, folds: int = 5) -> tuple:
        '''
        Finds the best hyperparameters for a model using sklearn's grid search and cross-validation. Works for both regressions and classification

        Inputs
        --------------
        model_class: The chosen model class (e.g., SGDRegressor())
        
        x_train: Training data
        
        y_train: Labels
        
        params_grid: Dictionary containing grid of values to test for hyperparameters

        scoring_metrics: List of all metrics you want to retrieve from the models.

        refit_metric: Metric used to refit the model

        folds: number of folds for cross-validation. Default value = 5

        Returns
        --------------
        best_model: Model instantitated with best hyperparameter values
        
        best_hyperparams: A dictionary containing optimal hyperparameters

        best_metrics_dict: A dictionary containing the best validation metrics
        '''
        model_instance = model_class # Create an instance of the model class
        grid_search = GridSearchCV(model_instance, param_grid = params_dict, scoring = scoring_metrics, refit = refit_metric, cv = folds)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_hyperparameter_dict = grid_search.best_params_

        # Retrieving dictionary of MSE and R2 below
        best_index = grid_search.best_index_
        best_metrics_dict = {}
        for var in scoring_metrics:
            best_metrics_dict[var] = grid_search.cv_results_[f'mean_test_{var}'][best_index]
        return best_model, best_hyperparameter_dict, best_metrics_dict

    def save_model(self, model_type : str , model_class: str,  best_model, best_hyperparameter: dict, best_metrics: dict) -> None: 
        '''
        Save model, hyperparameters and metrics after tuning

        Inputs:
        --------------
        model_type: can be either 'regression' or 'classification'
        
        model_class: can be any of the model classes. E.g., SGDRegressor()
        
        best_model: model that performs best along with its hyperparameters
        
        best_hyperparameter: best performing hyperparameter according to evaluation metric
        
        best_metrics: best validation metrics
        
        Returns:
        --------------
        Nothing
        
        '''
        # Check if model type is valid as it is used to make relevant sub-folders
        if model_type not in ['regression', 'classification']:
            raise ValueError("Invalid input_value. Please choose 'regression' or 'classification'.")

        load_dotenv() # Load environment variables from .env file
        models_folder = os.getenv(f'MODELS_DIR') # Imports directory path from .env file
        models_type_folder = os.path.join(models_folder, model_type)
        if os.path.exists(models_type_folder) == False:
            os.mkdir(models_type_folder)
        models_class_subfolder = os.path.join(models_type_folder, model_class)
        if os.path.exists(models_class_subfolder) == False:
            os.mkdir(models_class_subfolder)

        # Save the model, hyperparameters and metrics in the above folder
        joblib.dump(best_model, os.path.join(models_class_subfolder, 'model.joblib'))  
        with open(os.path.join(models_class_subfolder, 'hyperparameters.json'), 'w') as json_file: 
            json.dump(best_hyperparameter, json_file)
        with open(os.path.join(models_class_subfolder, 'metrics.json'), 'w') as json_file: 
            json.dump(best_metrics, json_file)

    def evaluate_all_models(self, model_type: str, model_class_and_hyperparameters: dict, X_train: pd.DataFrame, y_train: pd.Series,
                            scoring_metrics: list, refit_metric: str, folds: int = 5) -> None:
        '''
        Chooses optimal hyperparameters of each model and saves the best version of each model
        
        Inputs: 
        --------------
        model_type: Takes only two values - 'regression' or 'classification'

        model_class_and_hyperparameters: A dictionary of all model classes as keys and their associated values as dictionary of hyperparameter values for tuning
        
        X_train: training dataset
        
        y_train: labels for training

        scoring_metrics: List of all metrics you want to retrieve from the models

        refit_metric: Metric used to refit the model while tuning hyperparamaters
                
        Returns:
        --------------
        Nothing
        '''
        model_classes = model_class_and_hyperparameters.keys()

        # For each model class, tune the model with dictionary of hyperparameters
        for model_class in model_classes:
            print(f'Evaluating model class of {model_class}')
            print('----------------------------------------')
            params_dict = model_class_and_hyperparameters[model_class]
            best_model, best_hyperparameters, best_metrics = (
            self.tune_model_hyperparameters(model_class, X_train, y_train, params_dict = params_dict, 
                                    scoring_metrics = scoring_metrics, refit_metric = refit_metric,  folds = folds))

            # Save the best version of the model
            self.save_model(model_type = model_type, model_class = f'{model_class}', 
                    best_model = best_model, best_hyperparameter = best_hyperparameters, best_metrics = best_metrics)

    def find_best_model(self, model_type: str, model_class_and_hyperparameters: dict, scoring_metrics: list, optimisation_metric: str):
        '''
        Finds the best model out of all evaluated models with their tuned hyperparamaters
        
        Inputs: 
        --------------
        model_type: this can take either 'regression' or 'classification'

        model_class_and_hyperparameters: A dictionary of all model classes and their associated dictionary of hyperparameter values for tuning
        
        scoring_metrics: List containing all scoring metrics used for evaluating all models 

        optimisation_metric: The metric on which we will choose our best model           
        
        Returns:
        --------------
        Best model
        Associated dictionary of hyperparameters
        Dictionary of performance metrics
        '''
        
        # Checking if optimisation metric is within list of scoring metrics
        if optimisation_metric not in scoring_metrics:
            raise ValueError("Optimisation metric is not part of scoring metrics")
        # Initialise the variables of interest
        best_model = None
        best_model_hyperparameters = None
        best_model_metric = float('-inf')
        # Create the right folder to look into
        load_dotenv() 
        models_folder = os.getenv(f'MODELS_DIR') 
        models_subfolder = os.path.join(models_folder, model_type)

        for model_class in model_class_and_hyperparameters.keys():
            folder = os.path.join(models_subfolder, f'{model_class}')
            metrics_filepath = os.path.join(folder, 'metrics.json')
            # Load the metrics from the file
            with open(metrics_filepath, 'r') as file:
                all_metrics = json.load(file)
                key_metric = all_metrics[optimisation_metric]
            print(f'{model_class}: {key_metric}')
            if key_metric > best_model_metric:
                best_model_metric = key_metric
                best_model = model_class

        # Open the folder of the best model and load the saved model, hyperparameters and metrics
        best_model_folder = os.path.join(models_subfolder, f'{best_model}')
        model_path = os.path.join(best_model_folder, 'model.joblib')
        hyperparameters_path = os.path.join(best_model_folder, 'hyperparameters.json')
        metrics_path = os.path.join(best_model_folder, 'metrics.json')
        model = joblib.load(model_path)
        with open(hyperparameters_path, 'r') as hyperparams_file:
            hyperparameters = json.load(hyperparams_file)
        with open(metrics_path, 'r') as metrics_file:
            metrics = json.load(metrics_file)

        return model, hyperparameters, metrics


def load_and_standardise_numerical_data(label: str) -> tuple:
    '''
    Load data, split to train and test data, and standardise data

    Inputs: 
    --------------
    Label: variable needed as label

    Returns: 
    --------------
    X_train_scaled: Training data of features normalised using Z-score
    X_test_scaled: Test data of features normalised using Z-score
    y_train: Labels for training
    y_test: Labels for test

    '''
    # Load, split and standardise data
    X, y = load_airbnb_data(label)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_pd = pd.DataFrame(data = X_train_scaled, columns = X_train.columns)
    X_test_scaled_pd = pd.DataFrame(data = X_test_scaled, columns = X_test.columns)
    return X_train_scaled_pd, X_test_scaled_pd, y_train, y_test