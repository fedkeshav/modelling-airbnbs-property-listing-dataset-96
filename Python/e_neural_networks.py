#%%
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from a_tabular_data import load_airbnb_data
from b_modelling_functions import load_and_standardise_numerical_data
from sklearn.model_selection import train_test_split

# Create tensor Dataset and Dataloader
class AirbnbNightlyPriceRegressionDataset(Dataset):

    def __init__(self):
        self.features, self.labels = load_airbnb_data('Price_Night')
        self.data = pd.concat([self.features, self.labels], axis = 1)
        #self.features = torch.tensor(self.features.values, dtype = torch.float32)
        #self.labels = torch.tensor(self.labels.values, dtype = torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        features = torch.tensor(self.features.iloc[index]).float()
        label = torch.tensor(self.labels.iloc[index]).float()
        return (features, label)

dataset = AirbnbNightlyPriceRegressionDataset()
print(dataset[10])
len(dataset)
#%%
dataset_train, dataset_test = train_test_split(dataset.data, test_size=0.3)
dataset_validation, dataset_test = train_test_split(dataset_test, test_size=0.5)
print(len(dataset_train))
#%%

#%%
train_loader = DataLoader(dataset_train, batch_size= 29, shuffle=True)
print(len(train_loader))
next(iter(train_loader))
#%%
#%%
for batch in train_loader:
    print(batch)
    break
