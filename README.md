# Telco_Report

## Project Description:
The telecommunications enterprise Telco faces a significant challenge with customer retention. This project aims to unearth the underlying factors contributing to customer churn, employing advanced machine learning techniques to identify and predict patterns of customer attrition. The ultimate goal is to enable Telco to implement preemptive strategies to enhance customer loyalty and reduce churn, with the findings and predictive model being presented to the lead data scientist for strategic decision-making.

### Goals:
- Discover drivers of churn.
- Construct a machine learning classification model that can accurately predict churn.
- Present findings and process to lead data scientist, as well as recommendations.

## Initial Hypotheses:
- Churn is related to charges and potentially some services.

### Questions pertaining to the data:
- What is the relationship between monthly charges and churn?
- Does internet (or lack of) affect churn? 
- Do any of the internet services have a particular impact on churn?
- What contract types cause higher churn?

## Data Dictionary:
| Variable | Description |
| -------- | ----------- |
| index | Customer_ID, the identification code associated with each customer |
| gender | Male or Female, the gender of each customer |
| senior_citizen | Yes or No, whether or not the customer is a senior citizen |
| married | Yes or No, whether or not the customer has a spouse |
| children | Yes or No, whether or not the customer has children |
| tenure_months | Integer, the amount of months the customer has been with Telco |
| paperless_billing | Yes or No, whether a customer has signed up for paperless billing |
| monthly_charges | Float, each customer's monthly charge |
| total_charges | Float, each customer's total charge to current date |
| churn (target) | Yes or No, whether a customer has ended his or her contract with Telco |
| contract_type | The type of contract the customer has with Telco (Month-to-Month, 1 Year, or 2 Year) |
| internet_service_type | The type of internet the customer has with Telco (Fiber, DSL, or none) |
| payment_type | How a customer pays for his or her service |
| streaming | Details on which streaming services a customer may be subscribed to |
| phone_lines | Details on a customer's phone service |
| protection | Details on which online protection services a customer may have |
| support | Details on which online support services a customer may have |
| Additional features | Encoded values for categorical data for the sake of modeling |

## Project Pipeline:
- Acquisition
    - Acquire data from local CSV or Codeup's SQL database
- Preparation
    - Rename columns for clarification
        - married
        - children
        - tenure_months
    - Create engineered columns from existing data
        - streaming
        - phone_lines
        - protection
        - support
- Exploration
    - Explore the data to identify potential drivers of churn
        - What is the relationship between monthly charges and churn?
        - Does internet (or lack of) affect churn? 
        - Do any of the internet services have a particular impact on churn?
        - What contract types cause higher churn?
- Modeling
    - Encode the data
    - Use feature and hyperparameter selection to build predictive classification models
    - Test predictive models on train and validate
    - Isolate best model to run on test dataset
- Delivery
    - Draw conclusions
    - Present findings

## Reproduction of Findings:
1. Clone this repository
2. If you have access to the Codeup MySQL DB:
    - Save env.py in the repository with user, password, and host variables.
    - Ensure the env.py has the appropriate database connection.
    - RandomState 123 is predefined in the functions
    - Run the notebook.
3. If you don't have access:
    - Request access from Codeup.
    - Follow step 2 after obtaining access.

Notes: With the exception of acquire.py and env.py, all the .py files can be rebuilt from the contents of their respective notebooks.

## Key Findings

### Recommendations

### Takeaways