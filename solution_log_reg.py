import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt

train_data = './train.csv'
test_data = './test.csv'

# def eff_2_perc_improv_renewal_prob(x):
# 	return 20 * ( 1 - np.exp( -x/5 ) )

# def incentive_2_eff(x):
# 	return 10 * ( 1 - np.exp( -x/400 ) )

# def eff_2_incentive(x):
# 	return -400 * np.log(1+x/10)

# def slope_incent_vs_perc(x):
# 	return np.exp( -2*(1-np.exp(-x/400)) - x/400 )/10

# def slope_incent_vs_rev(x):
# 	return slope_incent_vs_perc(x) - 1

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


train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)

train_df = train_df.drop(['id'],axis=1)
test_ids = test_df.loc[:,'id'].values
test_ids = test_ids.reshape(-1,1)
test_premiums = test_df.loc[:,'premium'].values.reshape(-1,1).flatten()
test_df = test_df.drop(['id'],axis=1)

train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())

X,y = train_df.iloc[:,:-1], pd.DataFrame(train_df.iloc[:,-1])

# X['age'] = X['age_in_days'].divide(365)
# test_df['age'] = test_df['age_in_days'].divide(365)

# X['age'] = pd.cut(X['age'].values,3,labels=['young','middle_age','old'],retbins=True)[0]
# test_df['age'] = pd.cut(test_df['age'].values,3,labels=['young','middle_age','old'],retbins=True)[0]

# X = X.drop(['age_in_days'],axis = 1)
# test_df = test_df.drop(['age_in_days'],axis = 1)

conv2int_col_list = ['Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late']
X[conv2int_col_list] = X[conv2int_col_list].astype(np.int32)
test_df[conv2int_col_list] = test_df[conv2int_col_list].astype(np.int32)

# print(X.shape,test_df.shape)

categorical_cols = ['sourcing_channel','residence_area_type']

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

scaler = MinMaxScaler()
cols_that_need_to_be_scaled = ['Income','premium','age_in_days']
X[cols_that_need_to_be_scaled] = scaler.fit_transform(X[cols_that_need_to_be_scaled].values)
test_df[cols_that_need_to_be_scaled] = scaler.fit_transform(test_df[cols_that_need_to_be_scaled].values)

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


# X['incentives'] - create function for mapping incentive to effort, effort to improvement in probability
#figure out point beyond which ncrease in incentive will not increase probability of renewal

model = LogisticRegression(penalty='l2',C=1)
model = model.fit(X,y)
y_score = model.predict_proba(test_df)[:,1]
incentives = perc_improv_2_incentive(y_score,test_premiums)

# new_incentives = incentives.copy()

y_score = y_score.reshape( (y_score.shape[0],-1) )
incentives = incentives.reshape( (incentives.shape[0],-1) )

preds = np.concatenate([test_ids,y_score,incentives],axis=1)
preds_df = pd.DataFrame(data = preds, columns = ['id','renewal','incentives'])
preds_df.to_csv('prediction_log_reg_partb_incentives_max_rev.csv',index=False)


# incentive_old_df = pd.read_csv('prediction_log_reg_partb_incentives_max_rev.csv')
# inc_old = incentive_old_df['incentives'].values



# abc = np.not_equal(new_incentives,inc_old)

# diff = 0
# for i in range(abc.shape[0]):
# 	if abc[i] == True:
# 		d = (new_incentives[i] - inc_old[i])
# 		diff = diff + d

# print('TOTAL DIFFERENCE IN REVENUE: ',diff)



'''
--------------------------------------partB----------------------------------------------------
efforts = []
perc_imps = []
incs = []
slopes = []

inc_lim = 800

for i in range(inc_lim+1):
	eff = incentive_2_eff(i)
	perc_imp = eff_2_perc_improv_renewal_prob(eff)
	slope = slope_incent_vs_perc(i)
	efforts.append(eff)
	perc_imps.append(eff)
	incs.append(i)
	slopes.append(slope)


# X = np.array([1, 2, 3, 4, 5, 6, 7])
# Y = np.array([1.1,1.9,3.0,4.1,5.2,5.8,7])

X1 = np.array(incs)
# Y1 = np.array(efforts)
# X2 = Y1.copy()
Y1 = np.array(slopes)

print(np.min(Y1),np.max(Y1))
Y2 = np.array(perc_imps)
# plt.scatter (X,Y)
# slope, intercept = np.polyfit(X, Y, 1)
# print(slope)
# print(intercept)
# plt.plot(X, Y, 'r')
plt.plot(X1,Y2, 'r')

plt.plot(X1,Y1, 'b')
# plt.plot(X2,Y2, 'b')
plt.ylim(0.0,10.0)
plt.ylabel('PERC IMPROVEMENT(RED)/SLOPES(BLUE)')
plt.xlabel('INCENTIVES')
plt.xlim(0.0,inc_lim)
plt.show()

-----------------------------------------------------------------------------------------------
