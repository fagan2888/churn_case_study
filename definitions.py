import pandas as pd
import numpy as np
# model selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
# classifiers 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
# roc_curve from sklearn
from sklearn.metrics import roc_curve, roc_auc_score
# pretty pictures 
import matplotlib.pyplot as plt


## Creating confusion matrix from predicts and trues
def standard_confusion_matrix(y_true, y_predict):
    y_true, y_predict = y_true>0, y_predict>0

    tp, fp = (y_true & y_predict).sum(), (~y_true & y_predict).sum()
    fn, tn = (y_true & ~y_predict).sum(), (~y_true & ~y_predict).sum()
    return np.array([[tp, fp], [fn, tn]])


## Returns dictionary of threshold and assocated profit val for given cost_ben
def profit_curve(cost_ben, pred_probs, labels):
    order = pred_probs.argsort()[::0-1]	## list of indexes to sort in reverse order
    thresholds, profits = [], []

    #print(pred_probs)
    #pred_probs = pred_probs[:, 1]
    for ind in order:
        thresh = pred_probs[ind]	## Sets theshold to prob
        pos_class = pred_probs > thresh
        confusion_mat = standard_confusion_matrix(labels, pos_class)
        profit = (confusion_mat * cost_ben).reshape(1,-1).sum()/len(labels)
        
        profits.append(profit)
        thresholds.append(thresh)
    return (thresholds, profits)


####################################
#### DATA CLEANING AND HANDLING ####

## Gets Target Values from data frame
def get_target(df, days):
    df["last_trip_date"] = pd.to_datetime(df["last_trip_date"])
    cut_off = df['last_trip_date'].dt.date.max() -  pd.to_timedelta(days, unit='d')
    y_train = (df['last_trip_date'] < cut_off).astype(int)
    return y_train


## Cleaning the dataframe
def clean(df):
	## Dropping test row for analysis
    df = df.copy()
    df = df.drop(["last_trip_date"], 1)

    ## Filling missing values with average usage (useful to use -1 as well)
    df["avg_rating_by_driver"] = df.groupby("city").transform(lambda x: x.fillna(x.mean())).astype(float)
    df["avg_rating_of_driver"] = df.groupby("city").transform(lambda x: x.fillna(x.mean())).astype(float)

	## Getting dummies from cities and phone type
    cities = pd.get_dummies(df["city"])[["Astapor", "Winterfell"]]
    df = pd.concat([df, cities], 1)
    df = df.drop(["city"], 1)

    phones = pd.get_dummies(df["phone"])[["iPhone"]]
    df = pd.concat([df, phones], 1)
    df = df.drop(["phone"], 1)

    df['luxury_car_user'] = df['luxury_car_user'].astype(int)

    return df

########################################
#### TESTING MODELS ####################

## Gets cross validated accuracy and AUC for different models
def cross_val_accuracy(X_train, y_train, func):

	acc = sum(cross_val_score(func, X_train, y_train, scoring='accuracy'))/3
	roc = sum(cross_val_score(func, X_train, y_train, scoring='roc_auc'))/3

	func_name = str(func.__class__.__name__)

	print("{0:27} Train CV | Accuracy: {1:5.4} | ROC: {2:5.4}".format(func_name, acc, roc))
	return acc, roc


## Performing grid search on gradient boost to find best params
def gb_grid(X_train, y_train, X_test, y_test):
	gb_grid = {'max_depth': [3, None],
			  'loss': ['exponential', 'deviance'],
			  'learning_rate': [.1, .05, .01],
              'min_samples_split': [2, 4],
              'max_depth': [3, 5],
              'min_samples_leaf': [1, 2, 4],
              'n_estimators': [50, 100, 120],
              'random_state': [1]}

	rf_gridsearch = GridSearchCV(GradientBoostingClassifier(),
	                             gb_grid,
	                             n_jobs=-1,
	                             verbose=True,
	                             scoring='accuracy')
	rf_gridsearch.fit(X_train, y_train)

	print( "best parameters:", rf_gridsearch.best_params_ )

	best_rf_model = rf_gridsearch.best_estimator_
	preds = best_rf_model.predict(X_test)
	
	print("Accuracy of Best RF Model:", sum(preds == y_test)/len(y_test))


## Plotting profit curves with different cost benefit matrix's
def plot_profit(thresh, profs, title, sv):

	thresh_l = thresh[::20]
	profs_l = profs[::20]

	m_ind = thresh_l[profs_l.index(max(profs_l))]
	print("Max Threshold:", m_ind)

	plt.plot(thresh_l, profs_l, color='red', linestyle='--', label='Profit Curve')
	plt.axvline(m_ind, linestyle='--', color='blue', label='Max at:%.3f'%m_ind)
	plt.title(title)
	plt.xlabel("Probability Threshold")
	plt.ylabel("Profit")
	plt.legend()
	plt.grid()
	plt.savefig(sv)
	plt.show()


## Creating the profit curves and calling the plot function on them
def get_profit_plots(func, X_test, y_test):
	cost_ben = np.array([[90,-10],[-100,0]])
	title = "Profit vs Churn Probability Threshold\nBudget: Everyone Churns\nCost of Promo=10 | Cost of Churn=100"
	thresh, profs = profit_curve(cost_ben, func.predict_proba(X_test)[:,1], y_test)
	sv = "10_promo_100.jpg"
	plot_profit(thresh, profs, title,sv)

	cost_ben = np.array([[50,-150],[-200,0]])
	title = "Profit vs Churn Probability Threshold\nBudget: Everyone Churns\nCost of Promo=150 | Cost of Churn=200"
	thresh, profs = profit_curve(cost_ben, func.predict_proba(X_test)[:,1], y_test)
	sv = "150_promo_200.jpg"
	plot_profit(thresh, profs,title,sv)

	cost_ben = np.array([[25,-25],[-50,0]])
	title = "Profit vs Churn Probability Threshold\nBudget: Everyone Churns\nCost of Promo=25 | Cost of Churn=50"
	thresh, profs = profit_curve(cost_ben, func.predict_proba(X_test)[:,1], y_test)
	sv = "25_promo_50.jpg"
	plot_profit(thresh, profs,title,sv)


## Getting Final test data
def get_test_data(path):
    test_data = pd.read_csv(path)
    y_test = get_target(test_data, 30)
    
    test_clean = clean(test_data)
    test_clean.drop('signup_date', axis=1, inplace=True)
    
    return test_clean, y_test

## Plot of feature importances
def plot_feat_import(model, names):
    feature_importances = 100*model.feature_importances_ / np.sum(model.feature_importances_)
    feature_importances, feature_names, feature_idxs = zip(*sorted(zip(feature_importances, names, range(len(names)))))
    width = 0.8

    idx = np.arange(len(names))
    plt.barh(idx, feature_importances, align='center')
    plt.yticks(idx, feature_names)

    plt.title("Feature Importances in Gradient Booster")
    plt.xlabel('Relative Importance of Feature', fontsize=14)
    plt.ylabel('Feature Name', fontsize=14)
    plt.savefig("feature_importances.jpg")
    plt.tight_layout()
    plt.show()
