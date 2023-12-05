
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
    

def telco_plots(df):
    '''
    Simple function to plot some things to look at
    '''
    sns.countplot(data=df,x='contract_type',hue='churn')
    plt.title('How does contract type relate to churn?')
    plt.grid(alpha=0.4)
    plt.show()
    
    sns.histplot(data=df,x='monthly_charges',hue='churn',kde=True,multiple='stack',alpha=0.5,line_kws={'lw':2,'ls':'-.'})
    plt.title('Distribution of Monthly Charges in Relation to Churn')
    plt.grid(alpha=0.4)
    plt.show()
    
    sns.countplot(data=df,x='internet_service_type',hue='churn')
    plt.title('Number of Active Customers to Churned Customers According to Internet Type')
    plt.grid(alpha=0.4)
    plt.show()
    
    sns.catplot(data=df,x='contract_type',col='internet_service_type',hue='churn',kind='count')
    plt.suptitle('Comparison of Contract Type and Churn Split by Internet Service')
    plt.subplots_adjust(top=0.9)
    plt.show()
    
