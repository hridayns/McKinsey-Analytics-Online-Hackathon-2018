import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import xgboost as xgb

train_data = './train.csv'
test_data = './test.csv'

def calc_rev(perc_improv_in_p,p_bench,premium,incentive):
	return (((p_bench) * (1 + perc_improv_in_p) * premium) - incentive)

def rmse_err(x,y):
	return np.sqrt(np.sum(np.square(x-y))/y.shape[0])

def eff_2_incentive(x):
	return -400 * np.log(1-x/10)

def perc_improv_renewal_prob_2_eff(x):
	return -5 * np.log(1-x/20)

def perc_improv_2_incentive(p_bench,premium):
	perc_improv = (1 - p_bench)/p_bench * 100
	perc_improv = np.clip(perc_improv,0,17)
	best_inc = eff_2_incentive(perc_improv_renewal_prob_2_eff(perc_improv))
	max_rev = calc_rev(perc_improv,p_bench,premium,best_inc)

	for i in range(perc_improv.shape[0]):
		for j in range(int(perc_improv[i]),-1,-1):
			inc = eff_2_incentive(perc_improv_renewal_prob_2_eff(j))
			rev = calc_rev(j,p_bench[i],premium[i],inc)
			if rev > max_rev[i]:
				max_rev[i] = rev
				best_inc[i] = inc

	return best_inc

#IDEAS:

# TRY MAKING NEW FEATURE WITH PERC_OF_PREMIUM_CREDIT_CASH AND TOTAL PREMIUM FOR PERC_OF_PREMIUM_NOT_IN_CREDIT_CASH
# PERFORM PCA SOMEHOW
# TRY OTHER ML ALGORITHMS

# STUFF THAT WAS DONE:

# USE XGBOOST - GET XGBOOST TO HIGER AUC score - hyper parameter tuning
# ONEHOT ENCODE INCOME
# ONE HOT ENCODING INSTEAD OF LABEL ENCODING
# FINISH TEST DATA PIPELINE - RUN IT ON THE TEST AND CREATE CSV FOR SUBMISSION TO CHECK SCORE ON LEADERBOARD - ATLEAST ONCE
# USE DIFFERENT SCALER APART FROM MINMAX SCALER - Used StandardScaler (dint work)
# MAKE AGE INTO AGE RANGE AFTER DIVIDING BY 365 - Used to make three categories - young,middle_age,old - dint work out that well
# STRATEFIED K-FOLD CROSS VALIDATION SET
# FIND A WAY TO USE NEURAL NETWORKS WITH XGBOOST - NOT GONNA WORK

if __name__ == '__main__':
	train_df = pd.read_csv(train_data)
	test_df = pd.read_csv(test_data)

	train_df = train_df.drop(['id','application_underwriting_score'],axis=1)
	test_ids = test_df.loc[:,'id'].values
	test_ids = test_ids.reshape(-1,1)
	test_premiums = test_df.loc[:,'premium'].values.reshape(-1,1).flatten()
	test_df = test_df.drop(['id','application_underwriting_score'],axis=1)

	train_df = train_df.fillna(train_df.mean())
	test_df = test_df.fillna(test_df.mean())

	# print(train_df.describe())
	# print(test_df.describe())
	X,y = train_df.iloc[:,:-1], pd.DataFrame(train_df.iloc[:,-1])

	# age_splits = 5
	income_splits = 5

	# X['age'] = X['age_in_days'].divide(365)
	# test_df['age'] = test_df['age_in_days'].divide(365)

	# X['age'] = pd.cut(X['age'].values,age_splits,retbins=True)[0]
	# test_df['age'] = pd.cut(test_df['age'].values,age_splits,retbins=True)[0]

	# X = X.drop(['age_in_days'],axis = 1)
	# test_df = test_df.drop(['age_in_days'],axis = 1)

	X['Income'] = pd.cut(X['Income'].values,income_splits,retbins=True)[0]
	test_df['Income'] = pd.cut(test_df['Income'].values,income_splits,retbins=True)[0]

	conv2int_col_list = ['Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late']
	X[conv2int_col_list] = X[conv2int_col_list].astype(np.int32)
	test_df[conv2int_col_list] = test_df[conv2int_col_list].astype(np.int32)


	categorical_cols = ['sourcing_channel','residence_area_type','Income']

	le = LabelEncoder()
	for col in categorical_cols:
		tmp = pd.concat([X[col],test_df[col]])
		le.fit(tmp.values)
		X[col] = le.transform(X[col])
		test_df[col] = le.transform(test_df[col])

	ohe = OneHotEncoder(sparse=False)
	for col in categorical_cols:
		tmp = pd.concat([X[col],test_df[col]]).values.reshape(-1,1)
		ohe.fit(tmp)
		
		trans_tmp = ohe.transform(X[col].values.reshape(-1,1))
		tmp = pd.DataFrame(trans_tmp,columns=[col+'_'+str(i) for i in ohe.active_features_])
		X = pd.concat([X,tmp],axis=1).drop([col],axis=1)

		trans_tmp = ohe.transform(test_df[col].values.reshape(-1,1))
		tmp = pd.DataFrame(trans_tmp,columns=[col+'_'+str(i) for i in ohe.active_features_])
		test_df = pd.concat([test_df,tmp],axis=1).drop([col],axis=1)

	# print(df['age_in_days'].divide(365).head(5))

	scaler = StandardScaler()
	cols_that_need_to_be_scaled = ['premium','age_in_days','no_of_premiums_paid']#'application_underwriting_score','no_of_premiums_paid',
	X[cols_that_need_to_be_scaled] = scaler.fit_transform(X[cols_that_need_to_be_scaled].values)
	test_df[cols_that_need_to_be_scaled] = scaler.fit_transform(test_df[cols_that_need_to_be_scaled].values)


	# param_test1 = {
	#  'max_depth':range(3,10,2),
	#  'min_child_weight':range(1,6,2)
	# }

	# gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
	# 	min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
	# 	objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
	# 	param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
	# gsearch1.fit(X,y)
	# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
 	# {'max_depth': 3, 'min_child_weight': 3} 0.8177190468806927

	# param_test2 = {
 # 		'max_depth':[2,3,4],
 # 		'min_child_weight':[2,3,4]
 # 	}

	# gsearch2 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
	# 	min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
	# 	objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
	# 	param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
	# gsearch2.fit(X,y)
	# print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
 	# {'max_depth': 3, 'min_child_weight': 2} 0.8179112030004756

	# param_test3 = {
 # 		'gamma':[i/10.0 for i in range(0,5)]
 # 	}

	# gsearch3 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=3,
	# 	min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
	# 	objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
	# 	param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
	# gsearch3.fit(X,y)
	# print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
	# {'gamma': 0.4}] {'gamma': 0.0} 0.8179112030004756

	# param_test4 = {
 # 		'subsample':[i/10.0 for i in range(6,10)],
 # 		'colsample_bytree':[i/10.0 for i in range(6,10)]
 # 	}

	# gsearch4 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3,
	# 	min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
	# 	objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
	# 	param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
	# gsearch4.fit(X,y)
	# print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)
	# {'colsample_bytree': 0.6, 'subsample': 0.9} 0.818456600552192

	# param_test5 = {
 # 		'subsample':[i/100.0 for i in range(85,101,5)],
 # 		'colsample_bytree':[i/100.0 for i in range(55,70,5)]
 # 	}

	# gsearch5 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3,
	# 	min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
	# 	objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
	# 	param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
	# gsearch5.fit(X,y)
	# print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)
	# {'colsample_bytree': 0.55, 'subsample': 0.95} 0.818459570999762

	# param_test6 = {
 # 		'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
 # 		'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
 # 	}

	# gsearch6 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3,
	# 	min_child_weight=2, gamma=0.1, subsample=0.95, colsample_bytree=0.55,
	# 	objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
	# 	param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
	# gsearch6.fit(X,y)
	# print(gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_)
	 # {'reg_alpha': 5.0, 'reg_lambda': 5.0} 0.818924976201904

	# pca = PCA(n_components=X.shape[1])
	# pca.fit(X)
	# var= pca.explained_variance_ratio_
	# var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
	# print(var)
	# print(var1)

	# plt.plot(var1)
	# plt.show()
	# exit()

	pca = PCA(n_components=10) #11 when u dont make income categorical. ? when u add in no_of_premium for scaling
	pca.fit(X)
	pca.fit(test_df)
	pc_X = pca.fit_transform(X)
	pc_test = pca.fit_transform(test_df)

	# var= pca.explained_variance_ratio_
	# var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
	# print(pc.shape)
	# print(type(pc))
	# print(pca.components_)
	# print(X1)
	X = pd.DataFrame(data=pc_X,columns=['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'])#,'p11'])#,'p12'])
	test_df = pd.DataFrame(data=pc_test,columns=['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'])#,'p11'])#,'p12'])
	# exit()
	xgtrain = xgb.DMatrix(X, label=y)
	clf = xgb.XGBClassifier(
	                max_depth = 3,#3 - 10
	                n_estimators = 5000,
	                learning_rate = 0.01,#0.01 - 0.2
	                gamma = 0,
	                reg_lambda=5.0, 
	                reg_alpha=5.0, 
	                min_child_weight = 2, #1,2,3,4,...,10
					subsample=0.8,#0.5-1
					colsample_bytree=0.8,#0.5-1
					objective= 'binary:logistic',
					nthread=4,
					scale_pos_weight=1,
					seed=27
	                )
	xgb_param = clf.get_xgb_params()
	# do cross validation
	# print ('Start cross validation')
	cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=5, metrics=['auc'],early_stopping_rounds=50)
	# print('Best number of trees = {}'.format(cvresult.shape[0]))
	clf.set_params(n_estimators=cvresult.shape[0])
	# print('Fit on the training data')
	clf.fit(X, y, eval_metric='auc')
	print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X)[:,1]))#Overall AUC: 0.8273693499843329 -till paramtest6 (END)
	print('Accuracy:', accuracy_score(y, clf.predict(X)))#Accuracy: 0.940440559528133

# feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
# feat_imp.plot(kind='bar', title='Feature Importances')
# plt.ylabel('Feature Importance Score')


	pred = clf.predict_proba(test_df)

	incentives = perc_improv_2_incentive(pred[:,1],test_premiums)

	test_ids = test_ids.flatten()

	submission = pd.DataFrame({"id":test_ids, "renewal":pred[:,1], "incentives":incentives},columns = ['id','renewal','incentives'])
	submission.to_csv("xgboost_param_tuned_submission_std_pca.csv", index=False)




# print('Predict the probabilities based on features in the test set')
# pred = clf.predict_proba(sel_test, ntree_limit=cvresult.shape[0])

# submission = pd.DataFrame({"ID":test.index, "TARGET":pred[:,1]})
# submission.to_csv("submission.csv", index=False)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# model = xgb.XGBClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# accuracy = accuracy_score(y_test, predictions)
# print('MODEL ROC_AUC SCORE: ',roc_auc_score(y_test,y_pred))

# print("Accuracy: %.2f%%" % (accuracy * 100.0))



# exit()
# rmse_avg = 0
# k = 10
# skf = StratifiedKFold(n_splits=k, random_state=None)
# for train_index, val_index in skf.split(X,y): 
#     print("Train:", train_index.shape, "Validation:", val_index.shape) 
#     X_train, X_test = X.iloc[train_index], X.iloc[val_index] 
#     y_train, y_test = y.iloc[train_index], y.iloc[val_index]
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#     model = LogisticRegression(penalty='l2',C=1)
#     model = model.fit(X_train,y_train)
#     y_score = model.predict_proba(X_test)[:,1]
#     y_test = y_test.values.flatten()
#     print('MODEL ROC_AUC SCORE: ',roc_auc_score(y_test,y_score))
#     rmse = rmse_err(y_test,y_score)
#     rmse_avg = rmse_avg + rmse
#     print('MODEL RMSE: ',rmse)

# rmse_avg = rmse_avg/k
# print('AVG MODEL RMSE: ',rmse_avg)

'''
model = LogisticRegression(penalty='l2',C=1)
model = model.fit(X,y)
y_score = model.predict_proba(test_df)[:,1]
incentives = perc_improv_2_incentive(y_score,test_premiums)


y_score = y_score.reshape( (y_score.shape[0],-1) )
incentives = incentives.reshape( (incentives.shape[0],-1) )

preds = np.concatenate([test_ids,y_score,incentives],axis=1)
preds_df = pd.DataFrame(data = preds, columns = ['id','renewal','incentives'])
preds_df.to_csv('prediction_log_reg_partb_incentives_max_rev.csv',index=False)
'''