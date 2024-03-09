#%%
from a_tabular_data import load_airbnb_data
from b_modelling_functions import BestChoiceModels
from c_regressions import regression_hyperparams_dict
from dotenv import load_dotenv
from e_nn_regressions_price import get_data_loader
from e_nn_regressions_price import find_best_nn, import_split_scale_data_regression_NN, nn_hyperparams_dict
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#1. PREPARING DATA FOR ML AND NN MODELS
## First preparing the dataset generally
f,category = load_airbnb_data('Category')
le = LabelEncoder()
category2 = le.fit_transform(category)
category_series = pd.Series(category2, name = 'category')
features, labels = load_airbnb_data('bedrooms')
data = pd.concat([features, category_series, labels], axis = 1)
## Preparing the data specifically for machine learning models
train_data_scaled, validation_data_scaled, test_data_scaled = import_split_scale_data_regression_NN(data)
feature_train_scaled = train_data_scaled.iloc[:, :-1]
label_train = train_data_scaled.iloc[:, -1]
## Preparing the data specifically for neural network model
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]
train_loader, validation_loader, test_loader = get_data_loader(data)

#2. DEFINING HYPERPARAMS AND INPUTS 
## Hyperparams and inputs for ML models   
model_class_and_hyperparameters = regression_hyperparams_dict()
nn_hyperparams = nn_hyperparams_dict()
# Defining input folder where results can be saved
load_dotenv() 
storage_folder = os.getenv(f'BEDROOM_MODELS_DIR') 
#%%
#3a. Running ML Models to find the best one
if __name__ == "__main__":
    # Finding the best ML model
    reg_scoring_metrics = ['neg_mean_squared_error', 'r2']
    reg_refit_metric = 'neg_mean_squared_error'
    bcm = BestChoiceModels()
    bcm.evaluate_all_models(storage_folder = storage_folder, 
                            model_class_and_hyperparameters = model_class_and_hyperparameters,
                            X_train = feature_train_scaled, y_train = label_train,
                            scoring_metrics = reg_scoring_metrics,
                            refit_metric = reg_refit_metric, folds = 3)
    final_model, final_hyperparams, final_metrics = (
                bcm.find_best_model(storage_folder = storage_folder,
                                    model_class_and_hyperparameters = model_class_and_hyperparameters,
                                    scoring_metrics = reg_scoring_metrics,
                                    optimisation_metric = reg_refit_metric))
    print(f' Best ML model is {final_model}')
    print(f' Best ML hyperparameters are {final_hyperparams}')
    print(f' Best ML metrics are {final_metrics}')

#%%
#3b. Running NN models to find the best one
if __name__ == "__main__":
    # Finding best NN model
    input_size = features.shape[1]
    output_size = 1
    nn_model, nn_best_metrics, nn_best_hyperparams = find_best_nn(
                                            storage_folder,
                                            input_size, output_size, 
                                            nn_hyperparams, train_loader, 
                                            validation_loader, epochs = 10)
    print(f'NN_Metrics: {nn_best_metrics}')
    print(f'NN_Hyperparameters: {nn_best_hyperparams}')

# %%
#4. Best model is
if nn_best_metrics['average_last_epoch_validation_loss'] < abs(final_metrics['neg_mean_squared_error']):
    print(f'Best model is NN')
    print(f'NN_Metrics: {nn_best_metrics}')
    print(f'NN_Hyperparameters: {nn_best_hyperparams}')

else:
    print(f' Best model is {final_model}')
    print(f' Best ML hyperparameters are {final_hyperparams}')
    print(f' Best ML metrics are {final_metrics}')

######### RESULTS BELOW (9 March 15:30)
'''
Best model is SGDRegressor(alpha=0.05, penalty='elasticnet', random_state=42)
Best ML hyperparameters are {'alpha': 0.05, 'learning_rate': 'invscaling', 'loss': 'squared_error', 'max_iter': 1000, 'penalty': 'elasticnet', 'random_state': 42}
Best ML metrics are {'neg_mean_squared_error': -0.1809812497816866, 'r2': 0.8286861415734997}'''
# %%
