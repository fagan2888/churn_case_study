from definitions import *
from sklearn.model_selection import train_test_split, cross_val_score


## Main function with lots of commented out code for workflow
if __name__ == '__main__':
	# pull in train data
	train_churn = pd.read_csv('data/churn_train.csv')
	# set y
	y = get_target(train_churn, 30)
	# clean train data
	clean_train = clean(train_churn)
	# forget signup date
	clean_train.drop('signup_date', axis=1, inplace=True)
	# tag numbers
	X = clean_train.values
	# make the splits
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)


	### TESTING RANDOM FORREST VS GRADIENT BOOST ###

	######### RANDOM FORREST #########
	# rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1)
	# acc_rf, roc_rf = cross_val_accuracy(X_train, y_train, rf)


	######### GRADIENT BOOST #########
	# gb = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, random_state=1)
	# acc_rf, roc_rf = cross_val_accuracy(X_train, y_train, gb)


	### GRADIENT BOOST HIGHER ACCURACY ADN AUC .75 VS .69 ###
	#################################################

	### MOVING FORWARDS WITH GRADIENT BOOST ###

	###### GETTING PROFIT PLOTS #########
	# gdbr.fit(X_train, y_train)
	# get_profit_plots(gdbr, X_test, y_test)
	# acc_gb1, roc_gb1 = cross_val_accuracy(X_train, y_train, gdbr)


	###### PERFORMING GRID SEARCH #########
	#gb_grid(X_train, y_train, X_test, y_test)

	###### FINAL FIT WITH GRID SEARCH PARAMS #########
	## Done with params after gradient boost
	gdbr = GradientBoostingClassifier(learning_rate     =0.1, 
									  loss              ='exponential',
									  max_depth         =5,
									  min_samples_leaf  =2,
									  min_samples_split =2,
									  n_estimators      =100, 
									  random_state      =1)

	# try it on
	gdbr.fit(X, y)
	X_test_f, y_test_f = get_test_data('data/churn_test.csv')

	## Fitting on Final Test Data and Getting Accuracy
	preds = gdbr.predict(X_test_f)
	# how accurate?
	acc = (sum(preds == y_test_f) / len(y_test_f))
	# let's see 
	print("Final Model Accuracy of:", acc)


	## Creating ROC Plot and getting AUC
	preds_proba = gdbr.predict_proba(X_test_f).T[1]
	# score it
	auc = roc_auc_score(y_test_f, preds_proba)
	# see it
	print("Final Model AUC Score:", round(auc, 4))
	# curve and thresh
	fpr, tpr, thresholds = roc_curve(y_test_f, preds_proba)

	# set base
	plt.plot(fpr, tpr, '--')
	# title
	plt.title("ROC Plot")
	# x-axis 
	plt.xlabel("False Positive Rate")
	# y-axis
	plt.ylabel("True Positive Rate")
	# grid it
	plt.grid()
	# save it
	plt.savefig("images/roc.jpg")
	# show it
	plt.show()
