
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
    

def encode_df(df,target):
    '''
    Takes a processed dataframe and encodes the object columns for usage in modeling.
    
    Takes a dataframe and a target variable (assuming the target variable is an object). Target variable keeps the thing the model is being trained on from splitting and altering it.
    
    !!! MAKE ME MORE DYNAMIC !!!
    - Add functionality to check if passed a list or dataframe
    - If dataframe, then run standard loop
    - If list then check if each item is a dataframe (checking for train/validate/test)
    - If list and each item is dataframe, then try loop on each dataframe
    - Otherwise return an error
    '''
    # Get the object columns from the dataframe
    obj_col = [col for col in df.columns if df[col].dtype == 'O']
    
    # remove target variable
    obj_col.remove(target)
    
    # Begin encoding the object columns
    for col in obj_col:
        # Grab current column dummies
        dummies = pd.get_dummies(df[col],drop_first=True)
        
        # concatenate the names in a descriptive manner
        dummies.columns = [col+'_is_'+column for column in dummies.columns]

        # add these new columns to the dataframe
        for column in dummies.columns:
            df[column] = dummies[column].astype(float)
        
        # Drop the old columns from the dataframe
        df = df.drop(columns=col)
    
    return df    
    

def Xy_sets(tvt_set,target):
    '''
    Encodes and returns X_sets and y_sets. Takes a list of dataframes(train/validate/test). Iterates through each applying the encode_df function. Then splits the dataframes into the X and y sets.
    
    Requires X_set and y_set.
        X_set is a list of 3 dataframes with the target column dropped (should be in train, validate, test order)
        y_set is a list of 3 Series containing the target column that was dropped (should be in train, validate, test order)
    '''
    X_set = []
    y_set = []
    
    # encode and split into X and y
    # for the sets, 0 = train, 1 = validate, 2 = test (assuming that the tvt_set has been passed in proper order)
    for set_ in tvt_set:
        encoded_df = encode_df(set_,target)
    
        X_set.append(encoded_df.drop(columns=target))
        y_set.append(encoded_df[target])
    
    return X_set,y_set


def dt_modeling(X_set,y_set,r_parameter=123,n_models=20,plot=False):
    '''
    Returns a list of decision tree models with accuracy metrics. Can also plot the metrics for visualization.
    
    Requires X_set and y_set to run.
        - X_set is a list containing X_train,X_validate, and X_test. Only uses the first two.
        - y_set is a list containing y_train,y_validate, and y_test. Only uses the first two.
    Does not check for test sets.
    
    r_parameter is for setting random_state.
    n_models defines how many models to create.
    '''
    # create list to hold stats
    metrics = []

    for i in range(n_models):
        # print(i+1)
        model = DecisionTreeClassifier(max_depth=i+1,random_state=r_parameter)

        model.fit(X_set[0],y_set[0])

        output = {
            'type':'DecisionTree',
            'model':model,
            'train_acc':model.score(X_set[0],y_set[0]),
            'validate_acc':model.score(X_set[1],y_set[1]),
            'hyperparameters':'max_depth='+str(i+1)
        }

        metrics.append(output)

    dt_metrics = pd.DataFrame(metrics)
    dt_metrics['difference'] = dt_metrics.train_acc - dt_metrics.validate_acc
    dt_metrics['average'] = dt_metrics[['train_acc','validate_acc']].mean(axis=1)
    
    if plot==True:
        plt.figure(figsize=(8, 5))
        plt.plot(dt_metrics.index, dt_metrics.train_acc, label="Train", marker="o")
        plt.plot(dt_metrics.index, dt_metrics.validate_acc, label="Validate", marker="o")
        plt.fill_between(dt_metrics.index, dt_metrics.train_acc, dt_metrics.validate_acc, alpha=0.2)
        plt.xlabel("Model Number as Index in DF", fontsize=10)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title(f"Classification Model Performance: Decision Tree", fontsize=18)
        plt.legend(title="Scores", fontsize=12)
        plt.show()
    
    return dt_metrics


def rf_modeling(X_set,y_set,n_models=20,r_parameter=123,plot=False):
    '''
    Returns a list of random forest models with accuracy metrics. Can also plot the metrics for visualization.
    
    Requires X_set and y_set to run.
        - X_set is a list containing X_train,X_validate, and X_test. Only uses the first two.
        - y_set is a list containing y_train,y_validate, and y_test. Only uses the first two.
    Does not check for test sets.
    
    r_parameter is for setting random_state.
    n_models defines how many models to create.
    '''
    # loop random forest
    metrics = []

    for i in range(n_models):
        # print(i+1)
        model = RandomForestClassifier(max_depth=i+1,random_state=r_parameter)

        model.fit(X_set[0],y_set[0])

        output = {
            'type':'RandomForest',
            'model':model,
            'train_acc':model.score(X_set[0],y_set[0]),
            'validate_acc':model.score(X_set[1],y_set[1]),
            'hyperparameters':'max_depth='+str(i+1)
        }

        metrics.append(output)

    rf_metrics = pd.DataFrame(metrics)
    rf_metrics['difference'] = rf_metrics.train_acc - rf_metrics.validate_acc
    rf_metrics['average'] = rf_metrics[['train_acc','validate_acc']].mean(axis=1)
    
    if plot==True:
        plt.figure(figsize=(8, 5))
        plt.plot(rf_metrics.index, rf_metrics.train_acc, label="Train", marker="o")
        plt.plot(rf_metrics.index, rf_metrics.validate_acc, label="Validate", marker="o")
        plt.fill_between(rf_metrics.index, rf_metrics.train_acc, rf_metrics.validate_acc, alpha=0.2)
        plt.xlabel("Model Number as Index in DF", fontsize=10)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title(f"Classification Model Performance: Random Forest", fontsize=18)
        plt.legend(title="Scores", fontsize=12)
        plt.show()
    
    return rf_metrics


def knn_modeling(X_set,y_set,n_models=20,plot=False):
    '''
    Returns a list of K-Nearest Neighbors models with accuracy metrics. Can also plot the metrics for visualization.
    
    Requires X_set and y_set to run.
        - X_set is a list containing X_train,X_validate, and X_test. Only uses the first two.
        - y_set is a list containing y_train,y_validate, and y_test. Only uses the first two.
    Does not check for test sets.
    
    n_models defines how many models to create.
    '''
    # loop KNN
    metrics = []

    for i in range(n_models):
        # print(i+1)
        model = KNeighborsClassifier(n_neighbors=i+1)

        model.fit(X_set[0],y_set[0])

        output = {
            'type':'KNN',
            'model':model,
            'train_acc':model.score(np.ascontiguousarray(X_set[0]),y_set[0]),
            'validate_acc':model.score(np.ascontiguousarray(X_set[1]),y_set[1]),
            'hyperparameters':'n_neighbors='+str(i+1)
        }

        metrics.append(output)

    knn_metrics = pd.DataFrame(metrics)
    knn_metrics['difference'] = knn_metrics.train_acc - knn_metrics.validate_acc
    knn_metrics['average'] = knn_metrics[['train_acc','validate_acc']].mean(axis=1)

    if plot==True:
        plt.figure(figsize=(8, 5))
        plt.plot(knn_metrics.index, knn_metrics.train_acc, label="Train", marker="o")
        plt.plot(knn_metrics.index, knn_metrics.validate_acc, label="Validate", marker="o")
        plt.fill_between(knn_metrics.index, knn_metrics.train_acc, knn_metrics.validate_acc, alpha=0.2)
        plt.xlabel("Model Number as Index in DF", fontsize=10)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title(f"Classification Model Performance: K-Nearest Neighbor", fontsize=18)
        plt.legend(title="Scores", fontsize=12)
        plt.show()
    
    return knn_metrics


def lr_modeling(X_set,y_set,n_models=20,r_parameter=123,plot=False):
    '''
    Returns a list of Logistic Regression models with accuracy metrics. Can also plot the metrics for visualization.
    
    Requires X_set and y_set to run.
        - X_set is a list containing X_train,X_validate, and X_test. Only uses the first two.
        - y_set is a list containing y_train,y_validate, and y_test. Only uses the first two.
    Does not check for test sets.
    
    n_models defines how many models to create.
    r_parameter defines random_state hyperparameter
    '''
    # loop logistic regression
    metrics = []

    for i in range(n_models,0,-1):
        model = LogisticRegression(C=(i),random_state=r_parameter)

        model.fit(X_set[0],y_set[0])

        output = {
            'type':'LogisticRegression',
            'model':model,
            'train_acc':model.score(X_set[0],y_set[0]),
            'validate_acc':model.score(X_set[1],y_set[1]),
            'hyperparameters':'C='+str(i)
        }

        metrics.append(output)

    lr_metrics = pd.DataFrame(metrics)
    lr_metrics['difference'] = lr_metrics.train_acc - lr_metrics.validate_acc
    lr_metrics['average'] = lr_metrics[['train_acc','validate_acc']].mean(axis=1)
    
    if plot == True:
        plt.figure(figsize=(8, 5))
        plt.plot(lr_metrics.index, lr_metrics.train_acc, label="Train", marker="o")
        plt.plot(lr_metrics.index, lr_metrics.validate_acc, label="Validate", marker="o")
        plt.fill_between(lr_metrics.index, lr_metrics.train_acc, lr_metrics.validate_acc, alpha=0.2)
        plt.xlabel("Model Number as Index in DF", fontsize=10)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title(f"Classification Model Performance: Logistic Regression", fontsize=18)
        plt.legend(title="Scores", fontsize=12)
        plt.show()
    
    return lr_metrics


def metrics_filter(metric_df,y_train):
    '''
    Function to filter and select only the best models of the given algorithm.
    
    Accepts a dataframe of a metric and returns the dataframe limited to only the best three models.
    '''
    # get baseline
    baseline_acc = (y_train.mode()[0] == y_train).mean()
    # print(f'baseline: {baseline_acc}')
    
    # drop anything below baseline
    metric_df = metric_df[metric_df.validate_acc > baseline_acc]

    # drop anything with a difference bigger than 0.1
    # print(f'diff mean: {metric_df.difference.mean()}')
    metric_df = metric_df[metric_df.difference < metric_df.difference.mean()]

    # drop anything with an average less than average rounded to 1 decimal
    # print(f'avg mean: {metric_df.average.mean()}')
    metric_df = metric_df[metric_df.average >= metric_df.average.mean()]

    # drop rows that are less than or equal to the validate mean
    # print(f'val mean: {metric_df.validate_acc.mean()}')
    metric_df = metric_df[metric_df.validate_acc > metric_df.validate_acc.mean()]

    return metric_df.sort_values(['difference','validate_acc','train_acc'],ascending=[True,False,False])[:]


def final_models(models,val_cutoff=0.8,avg_cutoff=0.8):
    '''
    Takes a series of DataFrames containing models in the form of a list.
    
    Expects a list, returns a single DataFrame holding the top three models.
    '''
    # turn models into dataframe
    models = pd.concat(models,ignore_index=True)
    
    models = models[models.validate_acc > val_cutoff]
    models = models[models.average > avg_cutoff]
    
    return models[:3]

