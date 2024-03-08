#%%
from a_tabular_data import load_airbnb_data
from datetime import datetime
from dotenv import load_dotenv
import itertools
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchmetrics import R2Score
import yaml


class AirbnbNightlyPriceRegressionDataset(Dataset):
    '''
    This class inherits from Torch Dataset to create appropriate 
    dataset for running neural networks 
    '''
    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe
    
    def __len__(self) -> int:
        ''' Returns number of rows or length of the data'''
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        ''' Returns specific row of features and labels as a tuple'''
        features = torch.tensor(self.data.iloc[idx, :-1]).float()
        label = torch.tensor(self.data.iloc[idx, -1]).float()
        return (features, label)


def import_split_scale_data_regression_NN(data: pd.DataFrame) -> tuple:
    '''
    Imports data, splits data into train, validation and test sets, and standardises these datasets

    Inputs
    --------------
    Dataframe

    Returns
    --------------
    A tuple containing three dataframes - standarised train data, validation data and test data
    '''
    # Split the data
    train_data, test_data = train_test_split(data, test_size=0.3)
    validation_data, test_data = train_test_split(test_data, test_size=0.5)
    # Standardise training data on features, merge back with labels and convert to dataframe
    scaler = StandardScaler()
    train_feature_scaled = pd.DataFrame(scaler.fit_transform(train_data.iloc[:, :-1]), columns = train_data.columns[:-1])
    train_labels = train_data.iloc[:, -1]
    train_labels.index = train_feature_scaled.index # Need to do this as index of features dataframe changes once I convert it into dataframe above
    train_data_scaled = pd.concat([train_feature_scaled, train_labels], axis = 1)
    # Standardise validation data on features, merge back with labels and convert to dataframe
    validation_feature_scaled = pd.DataFrame(scaler.transform(validation_data.iloc[:, :-1]), columns = validation_data.columns[:-1])
    validation_labels = validation_data.iloc[:, -1]
    validation_labels.index = validation_feature_scaled.index
    validation_data_scaled = pd.concat([validation_feature_scaled, validation_labels], axis = 1)
    # Standardise test data on features, merge back with labels and convert to dataframe 
    test_feature_scaled = pd.DataFrame(scaler.transform(test_data.iloc[:, :-1]), columns = test_data.columns[:-1])
    test_labels = test_data.iloc[:, -1]
    test_labels.index = test_feature_scaled.index
    test_data_scaled = pd.concat([test_feature_scaled, test_labels], axis = 1)

    return train_data_scaled, validation_data_scaled, test_data_scaled

def get_dataset(data: pd.DataFrame) -> tuple:
    '''
    Converts dataframe into Torch Dataset object using the AirbnbNightlyPriceRegressionDataset class

    Inputs
    --------------
    Dataframe

    Returns
    --------------
    A tuple containing three Dataset objects - standarised train data, validation data and test data
    '''
    train_scaled_pd, validation_scaled_pd, test_scaled_pd = import_split_scale_data_regression_NN(data)
    train_dataset = AirbnbNightlyPriceRegressionDataset(train_scaled_pd)
    validation_dataset = AirbnbNightlyPriceRegressionDataset(validation_scaled_pd)
    test_dataset = AirbnbNightlyPriceRegressionDataset(test_scaled_pd)
    return train_dataset, validation_dataset, test_dataset

def get_data_loader(data: pd.DataFrame, batch_size: int = 29) -> tuple:
    '''
    Converts DataFrame into Dataloader object

    Inputs
    --------------
    data: Dataframe

    batch_size: The size of each batch for the dataloader object

    Returns
    --------------
    A tuple containing three Dataloader objects - standarised train data, validation data and test data
    '''
    train_dataset, validation_dataset, test_dataset = get_dataset(data)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    return train_loader, validation_loader, test_loader


class NN_complex(torch.nn.Module):
    ''' A class that defines how a Neural Network model will work'''
    def __init__(self, config: list, input_size: int, output_size: int):
        super().__init__()
        # Takes in four hyperparameters for configuration, along with input size and output size
        self.optimiser = config['optimiser']
        self.lr = config['lr']
        self.hidden_width = config['hidden_layer_width'] 
        self.depth = config['hidden_layer_depth']
        self.input_size = input_size # Number of features 
        self.output_size = output_size

        # Creating a dictionary with sequence of operations
        layers = []
        for x in range (self.depth):
            layers.append(torch.nn.Linear(self.input_size, self.hidden_width))
            layers.append(torch.nn.ReLU())
            self.input_size = self.hidden_width
        layers.append(torch.nn.Linear(self.input_size, self.output_size))
        self.layers = torch.nn.Sequential(*layers)

    def forward (self, features: pd.DataFrame):
        return self.layers(features)

 
def train(model: NN_complex, train_loader: DataLoader, validation_loader: DataLoader, epochs: int = 10) -> tuple:
    '''
    Trains the neural network model and returns various performance metrics for the model for both training and validation data

    Inputs
    --------------
    model: Instance of the NN_complex class.

    train_loader: DataLoader object containing training data

    validation_loader: DataLoader object containing validationd ata

    epochs: Number of times to train on the data

    Returns
    --------------
    A tuple containing six metrics
    avg_train_loss: Average MSE across all batches of last epoch for training data
    
    avg_train_r2: Average R2 across all batches of last epoch for training data
    
    avg_validation_loss: Average MSE across all batches of last epoch for validation data
    
    avg_validation_r2: Average R2 across all batches of last epoch for validation data

    average_epoch_training_duration: Average time it takes to train the model on training data in one epoch

    inference_latency: Average time taken to make a prediction on training data
    '''
    writer = SummaryWriter()
    batch_idx = 0
    training_time = 0
    pred_time = []

    # Set the optimiser based on the config
    if model.optimiser == 'SGD':
        optimiser = torch.optim.SGD(model.parameters(), lr = model.lr)
    elif model.optimiser == 'Adagrad':
        optimiser = torch.optim.Adagrad(model.parameters(), lr = model.lr)
    
    for epoch in range(epochs):
        # Training data
        total_train_loss = 0
        total_r2_train = 0
        start_time = time.time()
        batches_in_epoch = 0
        # For every batch of training data
        for batch in train_loader:
            features, labels = batch
            labels = labels.view(-1,1)
            start_pred = time.time()
            predictions = model(features)
            end_pred = time.time()
            elapsed_time = end_pred - start_pred
            pred_time.append(elapsed_time)
            loss = F.mse_loss(predictions, labels)
            total_train_loss += loss.item()
            r2_train = R2Score()
            r2_train = r2_train(predictions, labels)
            total_r2_train += r2_train
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar(f'Train_loss', loss.item(), batch_idx)
            batch_idx += 1
            batches_in_epoch += 1
        avg_train_loss = total_train_loss / batches_in_epoch
        avg_train_r2 = total_r2_train / batches_in_epoch
        avg_train_r2 = avg_train_r2.item()
        end_time = time.time()
        total_epoch_time = end_time - start_time
        training_time += total_epoch_time

        # Validation data
        total_val_loss = 0
        total_r2_val = 0
        num_batches = 0
        with torch.no_grad(): # No need to adjust gradient
            for batch in validation_loader:
                features, labels = batch
                labels = labels.view(-1,1)
                predictions = model(features)
                loss = F.mse_loss(predictions, labels)
                total_val_loss += loss.item()
                r2_val = R2Score()
                r2_val = r2_val(predictions, labels)
                total_r2_val += r2_val
                num_batches += 1
            avg_val_loss = total_val_loss / num_batches
            avg_val_r2 = total_r2_val / num_batches
            avg_val_r2 = avg_val_r2.item()
        
        print(f'Validation loss for {epoch} = {avg_val_loss}')
        writer.add_scalar(f'Validation_loss', avg_val_loss, epoch)
    
    average_epoch_training_duration = training_time / epochs
    inference_latency = sum(pred_time) / len(pred_time)
    
    return avg_train_loss, avg_train_r2, avg_val_loss, avg_val_r2, average_epoch_training_duration, inference_latency

def generate_nn_configs(hyperparam_dict: dict) -> list:
    '''
    Generates a list of dictionaries, where each dictionary is a hyperparameter configuration

    Inputs
    --------------
    None

    Returns
    --------------
    A list of dictionaries where each dictionary is a hyperparameter configuration
    '''
    keys, values = zip(*hyperparam_dict.items())
    configs = []
    for v in itertools.product(*values):
        configs.append(dict(zip(keys, v)))
    return configs

def save_model(subfolder: str,  model, best_hyperparameter: dict, best_metrics: dict) -> None: 
    '''
    Save model, hyperparameters and metrics after tuning

    Inputs:
    --------------
    subfolder: the folder where results need to be saved
    
    best_model: model that performs best along with its hyperparameters
    
    best_hyperparameter: best performing hyperparameter according to evaluation metric
    
    best_metrics: best validation metrics
    
    Returns:
    --------------
    Nothing
    
    '''
    # Save the model, hyperparameters and metrics in the above folder
    sd =  model.state_dict()
    torch.save(sd, os.path.join(subfolder, 'model.pt')) 
    with open(os.path.join(subfolder, 'hyperparameters.json'), 'w') as json_file: 
        json.dump(best_hyperparameter, json_file)
    with open(os.path.join(subfolder, 'metrics.json'), 'w') as json_file: 
        json.dump(best_metrics, json_file)

def find_best_nn(input_size: int, output_size: int, hyperparam_dict: dict, train_data: DataLoader, validation_data: DataLoader, epochs: int = 10) -> tuple:
    '''
    Finds the best neural network configuration which minimises validation loss. 
    Saves all hyperparameters and their results. 
    Also saves the best model with its hyperparameters and metrics

    Inputs
    --------------
    None

    Returns
    --------------
    The best model, its hyperparameters and performance metrics
    '''
    parameters_and_metrics = [] # A list which will contain all configs and metrics for each config
    best_val_loss = float('inf')
    configs = generate_nn_configs(hyperparam_dict) #Generate list of dictionaries (config) containing various configs as dictionary    
    # Train the model for each config
    config_id = 0
    for config in configs:
        model = NN_complex(config, input_size = input_size, output_size = output_size)
        (avg_train_loss, avg_train_r2, 
         avg_val_loss, avg_val_r2, 
        average_epoch_training_duration, inference_latency)  = train(model, train_data, validation_data, epochs)
        # Summarising all performance metrics into a dictionary
        performance_metrics = {'average_last_epoch_train_loss': avg_train_loss,
                               'average_last_epoch_train_r2': avg_train_r2,
                               'average_last_epoch_validation_loss': avg_val_loss,
                               'average_last_epoch_validation_r2': avg_val_r2,
                               'average_epoch_training_duration': average_epoch_training_duration,
                               'inference_latency': inference_latency}
        config_id += 1
        parameters_and_metrics.append(config_id)
        params = {'params': config} 
        parameters_and_metrics.append(params)
        metrics = {'performance_metrics': performance_metrics}
        parameters_and_metrics.append(metrics)
        parameters_and_metrics.append('''''''')
        parameters_and_metrics.append('''''''')

        # Finding the best config which minimises the validation loss on last epoch
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_metrics = performance_metrics
            best_hyperparams = config

    # Saving the best model, hyperparameters and associated metrics
    load_dotenv() # Load environment variables from .env file
    nn_folder = os.getenv(f'NN_MODELS_DIR') # Imports directory path from .env file
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    nn_time_folder = os.path.join(nn_folder, formatted_datetime)
    if os.path.exists(nn_time_folder) == False:
        os.mkdir(nn_time_folder)
    save_model(nn_time_folder, model, best_hyperparams, best_metrics)

    # Saving all configs and metrics associated in a single file
    with open(os.path.join(nn_time_folder, 'params_metrics.json'), 'w') as json_file:
        json.dump(parameters_and_metrics, json_file, indent = 6)

    return model, best_metrics, best_hyperparams

#%%
hyperparams_dict = {
    'optimiser' : ['SGD', 'Adagrad'],
    'lr' : [0.001, 0.01, 0.1],
    'hidden_layer_width' : [4, 7],
    'hidden_layer_depth' : [2, 4]
}

if __name__ == "__main__":
    features, labels = load_airbnb_data('Price_Night')
    data = pd.concat([features, labels], axis = 1)
    train_loader, validation_loader, test_loader = get_data_loader(data)
    input_size = features.shape[1]
    output_size = 1
    model, best_metrics, best_hyperparams = find_best_nn(input_size, output_size, hyperparams_dict,train_loader, validation_loader, epochs = 15)
    print(f'Metrics: {best_metrics}')
    print(f'Hyperparameters: {best_hyperparams}')