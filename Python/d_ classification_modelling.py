#%%
from b_modelling_functions import load_and_standardise_numerical_data, BestChoiceModels
from dotenv import load_dotenv
import os
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

def classification_hyperparams_dict():
    '''
    Creates a dictionary of all model classes and their associated hyperparameters for testing

    Inputs: 
    --------------
    None       
    
    Returns:
    --------------
    classification_model_class_and_hyperparameters: A dictionary containing each model class along with its associated
    dictionary of hyperparameters for testing
    '''

    # Define all the model classes and their hyperparameters to test    
    classification_model_classes = [LogisticRegression(),DecisionTreeClassifier(), 
                                    RandomForestClassifier(), GradientBoostingClassifier()]

    logreg_hyperparams_dict = {
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000],
        'multi_class': ['ovr'],
        'random_state': [42]
    }

    decision_tree_hyperparams_dict = {
        'ccp_alpha': [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
        'random_state': [42]
        }

    rf_hyperparams_dict = {
        'n_estimators': [100, 200],
        'max_features': [2 , 3],
        'ccp_alpha': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008],
        'random_state': [42]
        }

    gb_hyperparams_dict = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.2],
        'ccp_alpha': [0.0002, 0.0003, 0.0004],
        'subsample': [0.5, 1],
        'random_state': [42]
        }

    hyperparameters = [logreg_hyperparams_dict, decision_tree_hyperparams_dict,
                    rf_hyperparams_dict, gb_hyperparams_dict]
    classification_model_class_and_hyperparameters = dict(zip(classification_model_classes, hyperparameters))
    return classification_model_class_and_hyperparameters

# Defining model classes, hyperparameters, scoring and refit metrics
classification_model_class_and_hyperparameters = classification_hyperparams_dict()
class_scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
class_refit_metric = 'accuracy'
# Defining input folder where results can be saved
load_dotenv() 
storage_folder = os.getenv(f'CLAS_MODELS_DIR') 

#%%
if __name__ == '__main__':
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_standardise_numerical_data('Category')
    y_train_encoded, y_test_encoded = label_encoding(y_train, y_test)
    bcm = BestChoiceModels()
    bcm.evaluate_all_models(storage_folder = storage_folder,
                        model_class_and_hyperparameters= classification_model_class_and_hyperparameters,
                        X_train = X_train_scaled, y_train = y_train_encoded,
                        scoring_metrics = class_scoring_metrics,
                        refit_metric = class_refit_metric,
                        folds = 5)

    final_model, final_hyperparams, final_metrics = bcm.find_best_model(storage_folder = storage_folder, 
                                                                    model_class_and_hyperparameters = classification_model_class_and_hyperparameters, 
                                                                    scoring_metrics = class_scoring_metrics,
                                                                    optimisation_metric = class_refit_metric)
    print(f' Best model is {final_model}')
    print(f' Best hyperparameters are {final_hyperparams}')
    print(f' Best metrics are {final_metrics}')
#%%
# Deploying best model and hyperparameters on test data (better accuracy at 0.50)
from sklearn.metrics import accuracy_score
final_model.fit(X_train_scaled, y_train_encoded)
y_pred_encoded = final_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f'Accuracy: {accuracy}')
# %%
