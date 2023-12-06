
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
    

# Function to return whether answer on rejecting or failing to reject null hypothesis
def test_hypothesis(p, 
                    stat, 
                    tails='two', 
                    direction='greater',
                    alpha=0.05):
    '''
    test_hypothesis will take in a p value and a test statistic
    if p is less than a presumed alpha, then we  will reject
    our null hypothesis
    
    this takes in two positional arguments,
    p stat, a float value representing the probability of serendipity
    stat, a float value representing the test statistic
    
    with the keyword arguments f tails, direction, and alpha,
    the operator is able to change the control structrure in order
    to perform a one-tailed ttest if so desired
    '''
    if tails == 'two':
        if p < alpha:
            print(f'We reject the null hypothesis.\n Our p-value is {p} \n Our statistic value is {stat}')
        else:
            print(f'We fail to reject the null hypothesis.\n Our p-value is {p}.\n Our statistic is {stat}')
    else:
        if direction == 'greater':
            if ((p/2) < alpha) and (stat > 0):
                print(f'We reject our null hypothesis.\n Our p-value is {p} \n Our statistic value is {stat}')
            else:
                print(f'We fail to reject the null hypothesis.\n Our p-value is {p}.\n Our statistic is {stat}')
        else:
            if ((p/2) < alpha) and (stat < 0):
                print(f'We reject our null hypothesis.\n Our p-value is {p} \n Our statistic value is {stat}')
            else:
                print(f'We fail to reject the null hypothesis.\n Our p-value is {p}.\n Our statistic is {stat}')
    

def contract_churn(df):
    '''
    Function to compare churn against contract type. Takes a dataframe and returns nothing.
    '''
    # get statistics
    observed = pd.crosstab(df.churn,df.contract_type)
    stat,p,dof,expected = stats.chi2_contingency(observed)
    
    # create plot
    sns.countplot(data=df,x='contract_type',hue='churn')
    plt.title('How does contract type relate to churn?')
    plt.grid(alpha=0.4)
    plt.show()
    
    
    print('H_0: Churn is independent of contract type.')
    print('H_a: Churn is not independent of contract type.\n')
    
    test_hypothesis(p,stat)


def month_charges_churn(df):
    '''
    Function to compare churn against monthly charges. Takes a dataframe and returns nothing.
    '''
    
    sns.histplot(data=df,x='monthly_charges',hue='churn',kde=True,multiple='stack',alpha=0.5,line_kws={'lw':2,'ls':'-.'})
    plt.title('Distribution of Monthly Charges in Relation to Churn')
    plt.grid(alpha=0.4)
    plt.show()
    
    # test normal distribution
    stat,p = stats.shapiro(df.monthly_charges)
    print('Running Shapiro test for normalcy:')
    print('H_0: Monthly charges is distributed normally.')
    print('H_a: Monthly charges is not distributed normally.\n')
    test_hypothesis(p,stat)
    
    if p < 0.05:
        print()
        print('Running Mann-Whitney means test:')
        print('H_0: There is no difference between the monthly charges of customers who have churned and those who have not.')
        print('H_a: There is a difference between the monthly charges of customers who have churned and those who have not.\n')
        
        stat, p = stats.mannwhitneyu(
            df[df.churn == 'Yes'].monthly_charges,
            df[df.churn == 'No'].monthly_charges
        )
        
        test_hypothesis(p,stat)
    else:
        print()
        print("You're not supposed to see me...")


def internet_churn(df):
    '''
    Function to compare having internet or not against churn. Takes a dataframe and returns nothing.
    '''
    # isolate having internet vs no internet
    df['has_internet'] = np.where(df.internet_service_type != 'None','Has internet service','Has no internet')

    # plot having internet
    sns.countplot(df,x='has_internet',hue='churn')
    plt.title('Does Having Internet Affect Churn?')
    plt.grid(alpha=0.4)
    plt.show()

    # run stats test
    print('H_0: Churn and having internet are independent.')
    print('H_a: Churn and having internet are not independent.\n')

    observed = pd.crosstab(df.churn,df.has_internet)
    stat,p,dof,expected = stats.chi2_contingency(observed)
    test_hypothesis(p,stat)

    # drop column, because apparently it got added to the actual dataframe 
    df.drop(columns='has_internet',inplace=True)


def churn_contract_internet(df):
    '''
    Function to plot contract type and internet service. Takes a dataframe and returns nothing.
    '''
    
    # plot variables
    sns.catplot(data=df,x='contract_type',col='internet_service_type',hue='churn',kind='count')
    plt.suptitle('Comparison of Contract Type and Churn Split by Internet Service')
    plt.subplots_adjust(top=0.9)
    plt.show()


# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
    

# Function to return whether answer on rejecting or failing to reject null hypothesis
def test_hypothesis(p, 
                    stat, 
                    tails='two', 
                    direction='greater',
                    alpha=0.05):
    '''
    test_hypothesis will take in a p value and a test statistic
    if p is less than a presumed alpha, then we  will reject
    our null hypothesis
    
    this takes in two positional arguments,
    p stat, a float value representing the probability of serendipity
    stat, a float value representing the test statistic
    
    with the keyword arguments f tails, direction, and alpha,
    the operator is able to change the control structrure in order
    to perform a one-tailed ttest if so desired
    '''
    if tails == 'two':
        if p < alpha:
            print(f'We reject the null hypothesis.\n Our p-value is {p} \n Our statistic value is {stat}')
        else:
            print(f'We fail to reject the null hypothesis.\n Our p-value is {p}.\n Our statistic is {stat}')
    else:
        if direction == 'greater':
            if ((p/2) < alpha) and (stat > 0):
                print(f'We reject our null hypothesis.\n Our p-value is {p} \n Our statistic value is {stat}')
            else:
                print(f'We fail to reject the null hypothesis.\n Our p-value is {p}.\n Our statistic is {stat}')
        else:
            if ((p/2) < alpha) and (stat < 0):
                print(f'We reject our null hypothesis.\n Our p-value is {p} \n Our statistic value is {stat}')
            else:
                print(f'We fail to reject the null hypothesis.\n Our p-value is {p}.\n Our statistic is {stat}')
    

def contract_churn(df):
    '''
    Function to compare churn against contract type. Takes a dataframe and returns nothing.
    '''
    # get statistics
    observed = pd.crosstab(df.churn,df.contract_type)
    stat,p,dof,expected = stats.chi2_contingency(observed)
    
    # create plot
    sns.countplot(data=df,x='contract_type',hue='churn')
    plt.title('How does contract type relate to churn?')
    plt.grid(alpha=0.4)
    plt.show()
    
    
    print('H_0: Churn is independent of contract type.')
    print('H_a: Churn is not independent of contract type.\n')
    
    test_hypothesis(p,stat)


def month_charges_churn(df):
    '''
    Function to compare churn against monthly charges. Takes a dataframe and returns nothing.
    '''
    
    sns.histplot(data=df,x='monthly_charges',hue='churn',kde=True,multiple='stack',alpha=0.5,line_kws={'lw':2,'ls':'-.'})
    plt.title('Distribution of Monthly Charges in Relation to Churn')
    plt.grid(alpha=0.4)
    plt.show()
    
    # test normal distribution
    stat,p = stats.shapiro(df.monthly_charges)
    print('Running Shapiro test for normalcy:')
    print('H_0: Monthly charges is distributed normally.')
    print('H_a: Monthly charges is not distributed normally.\n')
    test_hypothesis(p,stat)
    
    if p < 0.05:
        print()
        print('Running Mann-Whitney means test:')
        print('H_0: There is no difference between the monthly charges of customers who have churned and those who have not.')
        print('H_a: There is a difference between the monthly charges of customers who have churned and those who have not.\n')
        
        stat, p = stats.mannwhitneyu(
            df[df.churn == 'Yes'].monthly_charges,
            df[df.churn == 'No'].monthly_charges
        )
        
        test_hypothesis(p,stat)
    else:
        print()
        print("You're not supposed to see me...")


def internet_churn(df):
    '''
    Function to compare having internet or not against churn. Takes a dataframe and returns nothing.
    '''
    # isolate having internet vs no internet
    df['has_internet'] = np.where(df.internet_service_type != 'None','Has internet service','Has no internet')

    # plot having internet
    sns.countplot(df,x='has_internet',hue='churn')
    plt.title('Does Having Internet Affect Churn?')
    plt.grid(alpha=0.4)
    plt.show()

    # run stats test
    print('H_0: Churn and having internet are independent.')
    print('H_a: Churn and having internet are not independent.\n')

    observed = pd.crosstab(df.churn,df.has_internet)
    stat,p,dof,expected = stats.chi2_contingency(observed)
    test_hypothesis(p,stat)

    # drop column, because apparently it got added to the actual dataframe 
    df.drop(columns='has_internet',inplace=True)


def churn_contract_internet(df):
    '''
    Function to plot contract type and internet service. Takes a dataframe and returns nothing.
    '''
    
    # plot variables
    sns.catplot(data=df,x='contract_type',col='internet_service_type',hue='churn',kind='count')
    plt.suptitle('Comparison of Contract Type and Churn Split by Internet Service')
    plt.subplots_adjust(top=0.9)
    plt.show()


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
    
