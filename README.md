# User Churn Case Study

This case study examines user data of a ride share company and attempts to predict when a user is at risk of churning. The following describes the structure of this repository and outlines some of the findings. A power point presentation on this study can be found [here](https://bit.ly/2H8KOgQ). The data is ommited from the repository as per request of the company.

### Structure:

- Initial EDA was done in eda.ipynb file
- The bulk of the model including data cleaning is done in churn_model.py
- Some helper functions for generating profit curves in in helpers.py
- Images directory contains
    - Profit curves with different budgets
    - Feature importance plot
    - ROC plot of our model

### Model:

- The final model used was a Gradient Boosted Model
- It performed slightly better than a Random Forrest and significantly better than a Logistic Regression model
- A Grid Search was run to find the optimized parameters
- Final Model had an AUC score of .845

### Findings:

- Several different budgets were created assuming that one way to get an at-risk customer to not churn was a promotion
    - We created several different budgets with this assumption in order to figure out what the optimal threshold would be in considering a user at-risk
    - These findings are displayed in the images folder
    - Analyzing the feature importance in our Gradient Boosted model revealed the the strongest factors were:
        - The % of time the user rode during surge pricing
        - The % of time the user used the service during the weekday
