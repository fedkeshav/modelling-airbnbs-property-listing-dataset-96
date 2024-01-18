#%%
import pandas as pd
import ast
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


class TabularData:

    def remove_rows_with_missing_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Removes rows that contain null values for ratings columns
        
        Input: Dataframe
        
        Returns:
            Dataframe
        '''
        rating_columns = [columns for columns in df.columns if 'rating' in columns]
        df.dropna(subset = rating_columns, axis =0, inplace=True, how ='any')
        return df

    def set_default_feature_values(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Replaces empty values with default of 1 for beds, bathrooms and bedrooms
        
        Input: Dataframe
        
        Returns:
            Dataframe
        '''
        columns = ['guests','bedrooms','bathrooms','beds']
        for var in columns:
            df[var].fillna(1, inplace=True)
        return df
    
    def parse_description_strings(self, description: str) -> str:
        '''
        Parses together elements of the Description column from a string of list items into a proper string

        Inputs:
            A string column
        
        Returns:
            A string column    
        '''
        try:
            # Parse description to be evaluated as list
            description_literal = ast.literal_eval(description)
            description_literal.pop(0)      # Remove 'About this space' section
            # Remove empty quotes from the list
            [description_literal.remove('') for quote in range(description_literal.count(''))]
            # Turn list into a string
            cleaned_description = ' '.join(description_literal)
            cleaned_description.replace('\n',' ')
            return cleaned_description
        except: # The error happens for one row which doesn't have any list type object. It simply saus Pool for 6 people
            return description

    def combine_description_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Converts description column from a list of strings in every row to a long string description
        
        Inputs: Dataframe
        
        Returns: 
            Dataframe
        '''
        df.dropna(subset = ['Description'], axis = 0, inplace=True)
        df.apply(self.parse_description_strings)
        return df

    def convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Converts guests and bedrooms column to int

        Inputs: Dataframe

        Returns:
            Dataframe with the two columns to int
        '''
        columns = ['guests','bedrooms']
        for var in columns:
            df[var] = pd.to_numeric(df[var], errors='coerce').astype('Int64')
        df['beds'] = df['beds'].astype('Int64')
        df.dropna(subset = ['guests', 'bedrooms'], how = 'any', axis = 0, inplace=True)
        return df
    
    def clean_tabular_data(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Cleans the dataframe

        Inputs: Dataframe
        
        Returns:
            Dataframe
        '''
        clean_df = self.remove_rows_with_missing_ratings(df)
        clean_df = self.combine_description_strings(clean_df)
        clean_df = self.set_default_feature_values(clean_df)
        clean_df = self.convert_to_numeric(clean_df)
        clean_df.drop('Unnamed: 19', axis = 1, inplace=True)
        clean_df.reset_index(drop=True, inplace=True)
        return clean_df
    

def import_data_to_df(directory_for_file: str, csv_filename:str) -> pd.DataFrame:
    '''
    Reads CSV file into a DataFrame
    
    Inputs:
        Full CSV file path and file name

    Returns:
        DataFrame
    '''
    df = pd.read_csv(f'{directory_for_file}/{csv_filename}')
    return df

def df_characteristics(df: pd.DataFrame) -> None:
    '''
    Prints important characteristics of the DataFrame
    Inputs:
        DataFrame
    Returns:
        Nothing
    '''
    df.info()
    df.head()

def export_to_csv (df: pd.DataFrame, directory_to_save: str, filename: str) -> str:
    '''
    Exports DataFrame to CSV
    
    Inputs: DataFrame and desired full path/name for the CSV file
    
    Returns:
        CSV file
    '''
    df.to_csv(f'{directory_to_save}/{filename}')

def load_airbnb_data(df: pd.DataFrame ) -> tuple:
    '''
    Returns a tuple containing feature columns and labels for data science

    Inputs: DataFrame, list of feature columns, and label column

    Returns:
        Tuple with dataframe for features and a series for label
    '''
    filter_df = df.drop('Price_Night', axis = 1)
    numeric_feature_columns = filter_df.select_dtypes(include=['number']).columns
    features = filter_df[numeric_feature_columns]
    labels = df['Price_Night']
    return (features, labels)

if __name__=="__main__":
    tabular = TabularData()
    directory = os.getenv("DATA_DIR") # Imports directory path from .env file
    df = import_data_to_df(directory, 'listing.csv')
    df_characteristics(df)
    clean_df = tabular.clean_tabular_data(df)
    export_to_csv(clean_df, directory, 'listings_clean.csv')
    df_characteristics(clean_df)

#%%
features, labels = load_airbnb_data(clean_df)
features.info()
print(labels)
# %%
