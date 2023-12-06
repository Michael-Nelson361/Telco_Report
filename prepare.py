
# import libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
    

# create function to prepare the data
def prep_telco(df):
    """
    Cleaning function to handle raw Telco dataset. Takes one argument (Telco Dataframe) and returns a cleaned and processed dataset.
    """
    drop_columns = [
        'phone_service',
        'multiple_lines',
        'streaming_movies',
        'streaming_tv',
        'payment_type_id',
        'internet_service_type_id',
        'contract_type_id',
        'online_security',
        'device_protection',
        'tech_support',
        'online_backup'
    ]
    
    # transform data
    df.senior_citizen = df.senior_citizen.map({0:'No',1:'Yes'})
    df.total_charges = df.total_charges.replace(' ',0).astype(float)
    df.internet_service_type = df.internet_service_type.fillna('None')
    df = df.set_index('customer_id')
    
    # combine columns
    df['streaming'] = np.select(
        [
        (df['streaming_movies'] == 'Yes') & (df['streaming_tv'] == 'Yes'),
        (df['streaming_movies'] == 'No') & (df['streaming_tv'] == 'No'),
        (df['streaming_movies'] == 'Yes'),
        (df['streaming_tv'] == 'Yes')
        ],
        ['Both', 'Neither', 'Movies', 'TV'],
        default='No internet service'
    )
    
    df['phone_lines'] = np.select(
        [
        (df['multiple_lines'] == 'No') & (df['phone_service'] == 'Yes'),
        (df['multiple_lines'] == 'Yes')
        ],
        ['Single', 'Multiple'],
        default='No phone service'
    )
    
    df['protection'] = np.select(
        [
        (df['online_security'] == 'Yes') & (df['device_protection'] == 'Yes'),
        (df['online_security'] == 'No') & (df['device_protection'] == 'No'),
        (df['online_security'] == 'Yes'),
        (df['device_protection'] == 'Yes')
        ],
        ['Both', 'Neither', 'Online Security', 'Device Protection'],
        default='No internet service'
    )
    
    df['support'] = np.select(
        [
        (df['tech_support'] == 'Yes') & (df['online_backup'] == 'Yes'),
        (df['tech_support'] == 'No') & (df['online_backup'] == 'No'),
        (df['tech_support'] == 'Yes'),
        (df['online_backup'] == 'Yes')
        ],
        ['Both', 'Neither', 'Tech Support', 'Online Backup'],
        default='No internet service'
    )
    
    # adjust column names
    df = df.rename(columns={
        'partner':'married',
        'dependents':'children',
        'tenure':'tenure_months'
    })
    
    # drop columns
    df = df.drop(columns=drop_columns)
    
    return df
    

def drop_extras(df,degree=6):
    """
    Function to drop extra columns that may have a smaller impact on the model. Requires dataframe be cleaned first.
    
    Takes a DataFrame, and returns a DataFrame.
    
    Degree indicates the index of object columns to begin selecting for drop off.
        Hint: smaller value means drop more columns, larger value means drop fewer columns!
    """
    from scipy import stats
    
    corr_dict = {}
    obj_cols = []
    alpha = 0.05
    
    # grab object columns from dataframe
    for col in df.columns:
        if df[col].dtype == 'O':
            # print(f'{col}: object')
            obj_cols.append(col)
    
    # get p-values of columns
    for col in obj_cols:
        observed = pd.crosstab(df.churn,df[col])
        # print(observed)

        t,p,dof,expected = stats.chi2_contingency(observed)

        # if p < alpha:
            # print(f'{col} has potential correlation with churn at {p}')
        corr_dict[col] = p
            
    
    # grabs 
    drop_extra = sorted(corr_dict, key=corr_dict.get)[degree:]
    
    df = df.drop(columns=drop_extra)
    
    return df
    

# Split given database
def split_df(df,strat_var,seed=123):
    """
    Returns three dataframes split from one for use in model training, validation, and testing. Takes two arguments:
        df: any dataframe to be split
        strat_var: the value to stratify on. This value should be a categorical variable.
    
    Function performs two splits, first to primarily make the training set, and the second to make the validate and test sets.
    """
    # Run first split
    train, validate_test = train_test_split(df,
                 train_size=0.60,
                random_state=seed,
                 stratify=df[strat_var]
                )
    
    # Run second split
    validate, test = train_test_split(validate_test,
                test_size=0.50,
                 random_state=seed,
                 stratify=validate_test[strat_var]
                )
    
    return train, validate, test
    

def drop_cols(df,cols=[],extras=False,degree=6):
    '''
    Drops columns. If no columns provided, then returns dataframe as is.
    
    Arguments:
    df: Required. DataFrame with columns to be dropped.
    cols: List, default is empty. If provided a list, then will drop the columns.
    extras: Default is False. If True, will run drop_extras function with provided degree.
        drop_extras will use a statistical test to determine a number of categorical columns to be dropped.
        Runs after other columns are dropped, which may impact the stats test run.
    degree: Default = 6. Used only in case extras is True.
    '''
    df = df.drop(columns=cols,errors='ignore')
    
    if extras == True:
        df = drop_extras(df,degree)
        
    return df
    
def drop_cols(df,cols=[],extras=False,degree=6):
    '''
    Drops columns. If no columns provided, then returns dataframe as is.
    
    Arguments:
    df: Required. DataFrame with columns to be dropped.
    cols: List, default is empty. If provided a list, then will drop the columns.
    extras: Default is False. If True, will run drop_extras function with provided degree.
        drop_extras will use a statistical test to determine a number of categorical columns to be dropped.
        Runs after other columns are dropped, which may impact the stats test run.
    degree: Default = 6. Used only in case extras is True.
    '''
    df = df.drop(columns=cols,errors='ignore')
    
    if extras == True:
        df = drop_extras(df,degree)
        
    return df
    
def drop_cols(df,cols=[],extras=False,degree=6):
    '''
    Drops columns. If no columns provided, then returns dataframe as is.
    
    Arguments:
    df: Required. DataFrame with columns to be dropped.
    cols: List, default is empty. If provided a list, then will drop the columns.
    extras: Default is False. If True, will run drop_extras function with provided degree.
        drop_extras will use a statistical test to determine a number of categorical columns to be dropped.
        Runs after other columns are dropped, which may impact the stats test run.
    degree: Default = 6. Used only in case extras is True.
    '''
    df = df.drop(columns=cols,errors='ignore')
    
    if extras == True:
        df = drop_extras(df,degree)
        
    return df
    
def drop_cols(df,cols=[],extras=False,degree=6):
    '''
    Drops columns. If no columns provided, then returns dataframe as is.
    
    Arguments:
    df: Required. DataFrame with columns to be dropped.
    cols: List, default is empty. If provided a list, then will drop the columns.
    extras: Default is False. If True, will run drop_extras function with provided degree.
        drop_extras will use a statistical test to determine a number of categorical columns to be dropped.
        Runs after other columns are dropped, which may impact the stats test run.
    degree: Default = 6. Used only in case extras is True.
    '''
    df = df.drop(columns=cols,errors='ignore')
    
    if extras == True:
        df = drop_extras(df,degree)
        
    return df
    