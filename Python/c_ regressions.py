#%%
# Importing relevant modules
from sklearn.linear_model import SGDRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from b_modelling_functions import load_and_standardise_numerical_data, BestChoiceModels
from sklearn.metrics import mean_squared_error
import pandas as pd

# Define all the model classes and their hyperparameters to test    
model_classes = [SGDRegressor(),DecisionTreeRegressor(), 
                 RandomForestRegressor(),
                 GradientBoostingRegressor()]

sgd_hyperparams_dict = {
    'loss': ['squared_error'],
    'alpha': [0.05,  0.1] ,
    'learning_rate': ['constant', 'optimal', 'invscaling'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'max_iter': [1000, 5000],
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
    'max_depth': [8, 12],
    'max_features': [2 , 3],
    'ccp_alpha': [100, 1000],
    'random_state': [42]
    }


gb_hyperparams_dict = {
    'loss': ['squared_error'],
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.2],
    'min_samples_split': [10, 20],
    'max_depth': [8, 12],
    'ccp_alpha': [100, 1000],
    'random_state': [42]
    }

hyperparameters = [sgd_hyperparams_dict, decision_tree_hyperparams_dict,
                   rf_hyperparams_dict, gb_hyperparams_dict]
model_class_and_hyperparameters = dict(zip(model_classes, hyperparameters))

# Defining scoring and refit metrics
model_type = 'regression'
reg_scoring_metrics = ['neg_mean_squared_error', 'r2']
reg_refit_metric = 'neg_mean_squared_error'

#%%
# Evaluate the best version of all models
if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_standardise_numerical_data('Price_Night')
    bcm = BestChoiceModels()
    bcm.evaluate_all_models(model_type = model_type, 
                            model_class_and_hyperparameters = model_class_and_hyperparameters,
                            X_train = X_train_scaled, y_train = y_train,
                            scoring_metrics = reg_scoring_metrics,
                            refit_metric = reg_refit_metric, folds = 3)
   
    final_model, final_hyperparams, final_metrics = (
                bcm.find_best_model(model_type = model_type,
                                    model_class_and_hyperparameters = model_class_and_hyperparameters,
                                    scoring_metrics = reg_scoring_metrics,
                                    optimisation_metric = reg_refit_metric))
    print(f' Best model is {final_model}')
    print(f' Best hyperparameters are {final_hyperparams}')
    print(f' Best metrics are {final_metrics}')
#%%
    # Fitting the best model to see weights and final predictions on test data
    final_model.fit(X_train_scaled, y_train)
    weights = final_model.coef_
    coef_data = pd.DataFrame({
        'Variable': X_train_scaled.columns,  # Assuming X is a DataFrame with column names
        'Coefficient': weights
    })
    print('')
    print('Weights of Coefficients')
    print(coef_data)

    y_pred = final_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean squared error: {mse}')
