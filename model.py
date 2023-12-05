
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import warnings
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
    

def encode_df(df):
    '''
    Takes a processed dataframe and encodes the object columns for usage in modeling.
    
    !!! MAKE ME MORE DYNAMIC !!!
    - Add functionality to check if passed a list or dataframe
    - If dataframe, then run standard loop
    - If list then check if each item is a dataframe (checking for train/validate/test)
    - If list and each item is dataframe, then try loop on each dataframe
    - Otherwise return an error
    '''
    # Get the object columns from the dataframe
    obj_col = [col for col in df.columns if df[col].dtype == 'O']
    
    # Begin encoding the object columns
    for col in obj_col:
        # Grab current column dummies
        dummies = pd.get_dummies(df[col],drop_first=True)
        
        # concatenate the names in a descriptive manner
        dummies.columns = [col+'_is_'+column for column in dummies.columns]

        # add these new columns to the dataframe
        for column in dummies.columns:
            df[column] = dummies[column]
        
        # Drop the old columns from the dataframe
        df = df.drop(columns=col)
    
    return df.astype(float)
    
