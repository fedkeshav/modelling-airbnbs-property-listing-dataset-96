# MODELLING AIRBNB PROPERTY LISTINGS

This project aims to make a few predictions using various models, including neural network.

The first prediction involves predicting prices of airbnb properly listings based on features of that listing such as ratings, number of bedrooms, number of bathrooms etc. We use both ML and Neural network models to do this.

The second prediction involves predicting property types based on its features such number of bedrooms, bathrooms etc. We use ML models for this.

The last prediction involves predicting number of bedrooms in a property. We used both ML and NN models for this.

## Dataset
The tabular dataset has the following columns. We did not use image data for the project.

    ID: Unique identifier for the listing
    Category: The category of the listing
    Title: The title of the listing
    Description: The description of the listing
    Amenities: The available amenities of the listing
    Location: The location of the listing
    guests: The number of guests that can be accommodated in the listing
    beds: The number of available beds in the listing
    bathrooms: The number of bathrooms in the listing
    Price_Night: The price per night of the listing
    Cleanliness_rate: The cleanliness rating of the listing
    Accuracy_rate: How accurate the description of the listing is, as reported by previous guests
    Location_rate: The rating of the location of the listing
    Check-in_rate: The rating of check-in process given by the host
    Value_rate: The rating of value given by the host
    amenities_count: The number of amenities in the listing
    url: The URL of the listing
    bedrooms: The number of bedrooms in the listing

## Folder structure

'models' folder: All model results can be found here

'Python': All python codes can be found here

'Raw data': All raw data (tabular and image) can be found here.

## USAGE INSTRUCTIONS
1. Download the raw data and save to your local directory.

2. Install the necessary environment including the packages and your local directory root path where you will be storing the raw data (see below installation instructions). Incorporate these paths in your local .env file.

3. Run 'a_tabular_data.py' to clean the Tabular data, and to retreive the numeric features set and a labels column.

4. Run 'b_modelling_functions.py' to get all necessary classes and functions to run regression and classification models.

5. Run 'c_regressions.py' to find the best model and hyperparameters for predicting property prices. See results under /models/regression

6. Run 'd_classifications.py' to find the best model and hyperparameters to classify each property into a category. Note the very low accuracy of c. 40%. See results under /models/classification

7. Run 'e_nn_regressions_price.py' to predict house prices using neural network model. This file indicates which hyperparameters yield the least validation loss. See results under /models/neural_networks

8. Run 'f_ML_NN_regressions_bedrooms.py' to predict number of bedrooms in a house using ML and NN models. See results under /models/bedroom_prediction

## KEY LEARNINGS
1. While tuning hyperparameters on decision trees, I kept getting errors for precision and recall being ill-defined for some hyperparameters. This was particularly for ccp_alpha hyperparameter. There are clever ways to figure out the range of hyperparameters to test for ccp_alpha using 'cost_complexity_pruning_path' attribute. Suggest taking similar approach for other tree based algorithms even if they don't have such an attribute. Plot accuracy againt ccp_alpha values.

2. While running neural networks, it is better to split datasets into train and validation, and then convert them to Torch datasets. Otherwise the split datasets convert from Torch Dataset to dataframes, which are not compatible with DataLoader class.

3. Always standardise datasets when using machine learning algorithms like SGD.

4. Same way we test for best ML model and hyperparameters, we should tetst between best ML model and best NN model.

5. Use of tensorboard for visualising loss curves - however, it can be challenging to visualise if you have many hyperparam setting (see example below)

![Alt Text](/Tensor%20visualiser.png)


## HOW TO INSTALL NECESSARY PACKAGES
1. Download the 'environment.yml' file to download all the packages needed to run the python files
2. Create the same environment from the downloaded 'environment.yml' file by using this code on your terminal: 'conda env create -f environment.yml'
3. Activate the environment using 'conda activate your_environment_name'

## KEY TECHNOLOGIES AND MODULES USED
1. sklearn packages
2. matplotlib
3. pandas
4. numpy
5. joblib
6. dotenv
7. PyTorch
8. TensorBoard

