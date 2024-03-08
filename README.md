# MODELLING AIRBNB PROPERTY LISTINGS

This project aims to make a few predictions using various models, including neural network.

The first prediction involves predicting prices of airbnb properly listings based on features of that listing such as ratings, number of bedrooms, number of bathrooms etc.

The second prediction involves predicting property types based on its features such number of bedrooms, bathrooms etc

## Folder structure

'models' folder: All model results can be found here

'Python': All python codes can be found here

'Raw data': All raw data (tabular and image) can be found here.

## USAGE INSTRUCTIONS
1. Download the raw data and save to your local directory.

2. Install the necessary environment including the packages and your local directory root path where you will be storing the raw data (see below installation instructions). Incorporate these paths in your local .env file.

3. Run 'a_tabular_data.py' to clean the Tabular data, and to retreive the numeric features set and a labels column.

4. Run 'b_modelling_functions.py' to get all necessary classes and functions to run regression and classification models.

5. Run 'c_regressions.py' to find the best model and hyperparameters for predicting property prices

6. Run 'd_classifications.py' to find the best model and hyperparameters to classify each property into a category. Note the very low accuracy of c. 40%.

7. Run 'e_nn_regressions_price.py' to predict house prices using neural network model. This file indicates which hyperparameters yield the least validation losss

## KEY LEARNINGS
1. While tuning hyperparameters on decision trees, I kept getting errors for precision and recall being ill-defined for some hyperparameters. This was particularly for ccp_alpha hyperparameter. There are clever ways to figure out the range of hyperparameters to test for ccp_alpha using 'cost_complexity_pruning_path' attribute. Suggest taking similar approach for other tree based algorithms even if they don't have such an attribute. Plot accuracy againt ccp_alpha values.

2. While running neural networks, it is better to split datasets into train and validation, and then convert them to Torch datasets. Otherwise the split datasets convert from Torch Dataset to dataframes, which are not compatible with DataLoader class.

3. Always standardise datasets when using machine learning algorithms like SGD.

## INSTALLATION INSTRUCTIONS

## KEY TECHNOLOGIES AND MODULES USED
1. sklearn packages
2. matplotlib
3. pandas
4. numpy
5. joblib
6. dotenv
7. PyTorch
8. TensorBoard

