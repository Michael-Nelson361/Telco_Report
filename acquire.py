
# import libraries
import pandas as pd
import numpy as np
import os
import env
        

def df_info(df,include=False,samples=1):
    """
    Function takes a dataframe and returns potentially relevant information about it (including a sample)

    include=bool, default to False. To add the results from a describe method, pass True to the argument.
    samples=int, default to 1. Shows 1 sample by default, but can be modified to include more samples if desired.
    """
    
    # create the df_inf dataframe
    df_inf = pd.DataFrame(index=df.columns,
            data = {
                'nunique':df.nunique()
                ,'dtypes':df.dtypes
                ,'isnull':df.isnull().sum()
            })
    
    # append samples based on input
    if samples >= 1:
        df_inf = df_inf.merge(df.sample(samples).iloc[0:samples].T,how='left',left_index=True,right_index=True)
    
    # append describe results if option selected
    if include == True:
        return df_inf.merge(df.describe(include='all').T,how='left',left_index=True,right_index=True)
    elif include == False:
        return df_inf
    else:
        print('Value passed to "include" argument is invalid.')
        

def print_libs():
    """
    Function that prints all libraries used up to present. Takes no arguments and returns none.
    """
    libraries = [
        'import pandas as pd',
        'import numpy as np',
        'import matplotlib.pyplot as plt',
        'import seaborn as sns',
        'from scipy import stats',
        'from pydataset import data',
        'import os',
        'import warnings',
        'from sklearn import metrics',
        'from sklearn.impute import SimpleImputer',
        'from sklearn.model_selection import train_test_split',
        'from sklearn.tree import DecisionTreeClassifier, plot_tree',
        'from sklearn.neighbors import KNeighborsClassifier',
        'from sklearn.ensemble import RandomForestClassifier',
        'from sklearn.linear_model import LogisticRegression'
    ]
    
    for library in libraries:
        print(library)
        

# Generic function to check if a file exists
def check_file_exists(filename,query,url):
    """
    Function takes a filename, query, and url and checks if the file exists. It will load the dataset requested from either SQL or from the local file.
    """
    if os.path.exists(filename):
        print('Reading from file...')
        df = pd.read_csv(filename,index_col=0)
    else:
        print('Reading from database...')
        df = pd.read_sql(query,url)
        
        df.to_csv(filename)
    
    return df
        

# Function to load telco dataset
def get_telco_data():
    """
    Function takes no arguments and returns a DataFrame containing the data from telco_churn database.
    
    This function requires an env file to be existent.
    """
    url = env.get_db_url('telco_churn')
    
    query = """
            select *
            from customers
            left join contract_types
                using(contract_type_id)
            left join internet_service_types
                using(internet_service_type_id)
            left join payment_types
                using(payment_type_id)
        """
    
    filename = 'telco.csv'
    
    # Import database
    telco_churn = check_file_exists(filename,query,url)
    
    return telco_churn


# Function to load telco dataset
def get_telco_data():
    """
    Function takes no arguments and returns a DataFrame containing the data from telco_churn database.
    
    This function requires an env file to be existent.
    """
    url = env.get_db_url('telco_churn')
    
    query = """
            select *
            from customers
            left join contract_types
                using(contract_type_id)
            left join internet_service_types
                using(internet_service_type_id)
            left join payment_types
                using(payment_type_id)
        """
    
    filename = 'telco.csv'
    
    # Import database
    telco_churn = check_file_exists(filename,query,url)
    
    return telco_churn

