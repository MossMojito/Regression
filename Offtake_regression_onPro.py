#!/usr/bin/env python
# coding: utf-8

# # Regression on Price Promotion data

# ## 1. Import Package and Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_path = 'Offtake_regression_input/offtake_regression_train.csv'
df_train = pd.read_csv(train_path)

df_train.tail()


# In[3]:


print(df_train.shape)


# ### Basic datafram checking

# In[4]:


df_train.columns 


# In[5]:


df_train.isnull().sum()


# In[6]:


df_train = df_train.dropna()


# In[7]:


df_train.info()


# ## 2. Deal with date

# In[8]:


df_train['date'] = pd.to_datetime(df_train['date'])
df_train['year'] = df_train['date'].dt.year
df_train['month'] = df_train['date'].dt.month
df_train['week_of_year'] = df_train['date'].dt.week


# In[9]:


df_train.head()


# ## 3. Deal with 'Pro'
# The Promotion ('pro') are promotions in each time. uniques pro in our data was show below

# again plot 'sales_in' and not line was perform better than before.

# In[10]:


sns.countplot(x=df_train["pro"])


# In[11]:


df_train['pro'].unique()


# In[12]:


df_train = df_train[df_train['pro'].isin(['RSP', '2for', 'point', '2nd_haft','3for'])]


# In[13]:


df_train['pro'].unique()


# in this section, author was decided to cut havvy promotion such as BOGO (Buy one ge one) and buy2+1 off and focus on the price discount promotion.( RSP is normal retail selling price, it mean no promotion) and next we were going to get dummy of 'Pro'

# In[14]:


dummies = df_train[['pro']]
dummies_col = dummies.columns


# In[15]:


df_train = pd.get_dummies(df_train, columns = dummies_col)
df_train = df_train[['date','year','month','week_of_year','sales_in','stores','pro_point',
         'pro_2for','pro_2nd_haft','pro_3for','pro_RSP','offtake']]
df_train.head()


# to prevent name confusing of pro_RSP back to RSP (no promotion)

# In[16]:


df_train = df_train.rename(columns = {'pro_RSP' : 'RSP'})


# now we complete df to afer deal with date, remove outlier and get dummy of categorical feature.

# ## 4. Visualization 

# In[17]:


fig, ax = plt.subplots(figsize=(10, 3))
sns.lineplot(data= df_train, x ='date', y= 'offtake', ax=ax, color='r')


# In[18]:


fig, ax = plt.subplots(figsize=(10, 3))
sns.lineplot(data= df_train, x ='date', y= 'stores', ax=ax)


# In[19]:


fig, ax = plt.subplots(figsize=(10, 3))
sns.lineplot(data= df_train, x ='date', y= 'sales_in', ax=ax)


# We can see aggresive peak of 'sales_in' at around 4000 - 4300, so author was decided to call those were outlier that need to remove. 

# ### remove outlier

# In[20]:


df_train.sort_values(by ='sales_in', ascending=False)


# In[21]:


outlier1 = df_train[(df_train['sales_in'] == 4331)].index
outlier2 = df_train[(df_train['sales_in'] == 4170)].index


# In[22]:


df_train.drop(outlier1, inplace = True)


# In[23]:


df_train.drop(outlier2, inplace = True)


# In[24]:


df_train.sort_values(by ='sales_in', ascending=False)


# In[25]:


fig, ax = plt.subplots(figsize=(10, 3))
sns.lineplot(data= df_train, x ='date', y= 'sales_in', ax=ax)


# # Regression

# ## 1. Features correlation and selection

# In[26]:


obs = df_train.shape[0]
types = df_train.dtypes
counts = df_train.apply(lambda x: x.count())
uniques = df_train.apply(lambda x: [x.unique()]).transpose()
nulls = df_train.apply(lambda x: x.isnull().sum())
distincts = df_train.apply(lambda x: x.unique().shape[0])
missing_ration = (df_train.isnull().sum()/ obs) * 100
skewness = df_train.skew()
kurtosis = df_train.kurt() 


# In[27]:


corr =df_train.corr()['offtake']
corr_col = 'corr '  + 'offtake'
str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)
cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]
str.columns = cols
str.sort_values(by ='corr offtake', ascending=False)


# In[28]:


df_corr = df_train.corr().abs()


# In[29]:


plt.figure(figsize=(12,8))
mask = np.triu(np.ones_like(df_corr, dtype=bool))
sns.heatmap(df_corr, annot=True, fmt=".2f", linewidths=.5, mask=mask, robust=True)


# In[30]:


sns.set(font_scale=0.75)
g = sns.pairplot(df_train[['month', 'week_of_year','sales_in', 'stores','pro_2for','pro_3for','RSP', 'offtake']])


# As heatmap it show correlation between each feature with the target (offtake) and double inspect with pairplot. In the end, Author set 0.2 as citeria to select freture 

# In[31]:


features_columns = ['week_of_year','sales_in', 'stores','pro_2for','pro_3for','RSP']


# ## 2. Preprocessing

# In[32]:


len(df_train)


# In[33]:


train = df_train[:90]
valid = df_train[90:]


# In[34]:


plt.figure(figsize=(20, 2))
sns.lineplot(data=train, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=valid, x="date", y="offtake", marker='o', linestyle='')


# In[35]:


plt.figure(figsize=(15, 2));
sns.distplot(train["offtake"]);
sns.distplot(valid["offtake"]);
plt.legend(["train", "valid"])


# In[36]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[37]:


norm = MinMaxScaler()
stand = StandardScaler()


# In[38]:


X_trian = train[features_columns].values
X_val = valid[features_columns].values

y_train = train["offtake"].tolist()
y_val = valid["offtake"].tolist()


# In[39]:


X_trian.shape, X_val.shape


# In[40]:


X_trian[0:5]


# In[41]:


X_train_norm = norm.fit_transform(X_trian)
X_train_std = stand.fit_transform(X_train_norm)

X_val_norm = norm.transform(X_val)
X_val_std = stand.transform(X_val_norm)


# In[42]:


X_train_norm[0:5]


# In[43]:


X_train_std[0:5]


# In[44]:


plt.figure(figsize=(15, 2));
sns.distplot(X_train_std);
sns.distplot(X_val_std);
plt.legend(["train", "valid"])


# ## 3. Modeling & Evaluation

# ### 3.1 baseline with **DecisionTreeRegressor**

# In[45]:


from sklearn.tree import DecisionTreeRegressor
import sklearn
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV


# In[46]:


param_grid = {'max_depth': [5, 6, 7, 8, 9, 10, 11, 12]}


# In[47]:


clf_tree = DecisionTreeRegressor(random_state=2022)


# In[48]:


grid = GridSearchCV(estimator=clf_tree, 
                    param_grid=param_grid, 
                    scoring="neg_mean_squared_error", 
                    n_jobs=-1)

grid.fit(X_train_std, y_train)


# In[49]:


print(grid.best_score_)
print(grid.best_estimator_.max_depth)


# In[50]:


opt_clf_tree = DecisionTreeRegressor(max_depth=7, random_state=2022)
opt_clf_tree.fit(X_train_std, y_train)


# In[51]:


y_train_opt_clf_tree = opt_clf_tree.predict(X_train_std)
y_val_opt_clf_tree = opt_clf_tree.predict(X_val_std)


# In[52]:


print("R-Square Train: ", r2_score(y_train, y_train_opt_clf_tree))
print("R-Square Validation: ", r2_score(y_val, y_val_opt_clf_tree))
print("="*50)
print("MAE Train: ", mean_absolute_error(y_train, y_train_opt_clf_tree))
print("MAE Validation: ", mean_absolute_error(y_val, y_val_opt_clf_tree))
print("="*50)


# In[53]:


train["baseline"] = y_train_opt_clf_tree
valid["baseline"] = y_val_opt_clf_tree


# In[54]:


plt.figure(figsize=(20, 2))
sns.lineplot(data=train, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=train, x="date", y="baseline", marker='o', linestyle='')
plt.legend(["Train", "Predict"])

plt.figure(figsize=(20, 2))
sns.lineplot(data=valid, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=valid, x="date", y="baseline", marker='o', linestyle='')
plt.legend(["Valid", "Predict"])


# ### 3.2 baseline with DecisionTreeRegressor + **Validation** (Timeseriessplit)

# In[55]:


from sklearn.model_selection import TimeSeriesSplit


# In[56]:


tscv = TimeSeriesSplit(n_splits=5, test_size=10)

cv_ls = []
for train_index, valid_index in tscv.split(X_train_std):
    cv_ls.append((train_index, valid_index))


# In[57]:


tscv


# In[58]:


cv_ls[0]


# In[59]:


cv_ls[1]


# In[60]:


plt.figure(figsize=(20, 2))
sns.lineplot(data=train.iloc[cv_ls[0][0]], x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=train.iloc[cv_ls[0][1]], x="date", y="offtake", marker='o', linestyle='')
plt.legend(["Train", "Predict"])

plt.figure(figsize=(20, 2))
sns.lineplot(data=train.iloc[cv_ls[1][0]], x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=train.iloc[cv_ls[1][1]], x="date", y="offtake", marker='o', linestyle='')
plt.legend(["Train", "Predict"])

plt.figure(figsize=(20, 2))
sns.lineplot(data=train.iloc[cv_ls[2][0]], x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=train.iloc[cv_ls[2][1]], x="date", y="offtake", marker='o', linestyle='')
plt.legend(["Train", "Predict"])


# In[61]:


clf_tree = DecisionTreeRegressor(random_state=2022)


# In[62]:


grid = GridSearchCV(estimator=clf_tree, 
                    param_grid=param_grid, 
                    scoring="neg_mean_squared_error", 
                    cv=cv_ls,
                    n_jobs=-1)

grid.fit(X_train_std, y_train)


# In[63]:


print(grid.best_score_)
print(grid.best_estimator_.max_depth)


# In[64]:


opt_clf_tree = DecisionTreeRegressor(max_depth=8, random_state=2022)
opt_clf_tree.fit(X_train_std, y_train)


# In[65]:


y_train_opt_clf_tree = opt_clf_tree.predict(X_train_std)
y_val_opt_clf_tree = opt_clf_tree.predict(X_val_std)


# In[66]:


print("R-Square Train: ", r2_score(y_train, y_train_opt_clf_tree))
print("R-Square Validation: ", r2_score(y_val, y_val_opt_clf_tree))
print("="*50)
print("MAE Train: ", mean_absolute_error(y_train, y_train_opt_clf_tree))
print("MAE Validation: ", mean_absolute_error(y_val, y_val_opt_clf_tree))
print("="*50)


# In[67]:


train["validate"] = y_train_opt_clf_tree
valid["validate"] = y_val_opt_clf_tree


# In[68]:


plt.figure(figsize=(20, 2))
sns.lineplot(data=train, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=train, x="date", y="validate", marker='o', linestyle='')
plt.legend(["Train", "Predict"])

plt.figure(figsize=(20, 2))
sns.lineplot(data=valid, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=valid, x="date", y="validate", marker='o', linestyle='')
plt.legend(["Valid", "Predict"])


# ### 3.3 baseline with DecisionTreeRegressor + Validation (Timeseriessplit) + **more param** 

# In[69]:


clf_tree = DecisionTreeRegressor(random_state=2022)


# In[70]:


param_grid = {'max_depth': [5, 6, 7, 8, 9, 10, 11, 12],
              'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
              'max_features': ["auto", "sqrt", "log2"]}


# In[71]:


grid = GridSearchCV(estimator=clf_tree, 
                    param_grid=param_grid, 
                    scoring="neg_mean_squared_error", 
                    cv=cv_ls,
                    n_jobs=-1)

grid.fit(X_train_std, y_train)


# In[72]:


print(grid.best_score_)
print(grid.best_estimator_)


# In[73]:


opt_clf_tree = DecisionTreeRegressor(random_state=2022, max_depth=6, max_features='sqrt', min_samples_split=8)
opt_clf_tree.fit(X_train_std, y_train)


# In[74]:


y_train_opt_clf_tree = opt_clf_tree.predict(X_train_std)
y_val_opt_clf_tree = opt_clf_tree.predict(X_val_std)


# In[75]:


print("R-Square Train: ", r2_score(y_train, y_train_opt_clf_tree))
print("R-Square Validation: ", r2_score(y_val, y_val_opt_clf_tree))
print("="*50)
print("MAE Train: ", mean_absolute_error(y_train, y_train_opt_clf_tree))
print("MAE Validation: ", mean_absolute_error(y_val, y_val_opt_clf_tree))
print("="*50)


# In[76]:


train["tree_param_grid"] = y_train_opt_clf_tree
valid["tree_param_grid"] = y_val_opt_clf_tree


# In[77]:


plt.figure(figsize=(20, 2))
sns.lineplot(data=train, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=train, x="date", y="tree_param_grid", marker='o', linestyle='')
plt.legend(["Train", "Predict"])

plt.figure(figsize=(20, 2))
sns.lineplot(data=valid, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=valid, x="date", y="tree_param_grid", marker='o', linestyle='')
plt.legend(["Valid", "Predict"])


# ### 3.4 **RandomForestRegressor** + Validation (Timeseriessplit) + more param 

# In[78]:


from sklearn.ensemble import RandomForestRegressor


# In[79]:


clf_rand = RandomForestRegressor(random_state=2022)


# In[80]:


param_grid = {
    'bootstrap': [True, False],
    'n_estimators': [100],
    'max_depth': [5, 6, 7, 8, 9, 10, 11, 12],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_features': ["auto", "sqrt", "log2"]}


# In[81]:


grid = GridSearchCV(estimator=clf_rand, 
                    param_grid=param_grid, 
                    scoring="neg_mean_squared_error", 
                    cv=cv_ls,
                    n_jobs=-1)

grid.fit(X_train_std, y_train)


# In[82]:


print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)


# In[83]:


opt_clf_rand = RandomForestRegressor(random_state=2022, bootstrap= True, min_samples_leaf = 1, n_estimators= 100, max_depth=5, max_features='sqrt', min_samples_split=10)
opt_clf_rand.fit(X_train_std, y_train)


# In[84]:


y_train_opt_clf_rand = opt_clf_rand.predict(X_train_std)
y_val_opt_clf_rand = opt_clf_rand.predict(X_val_std)


# In[85]:


print("R-Square Train: ", r2_score(y_train, y_train_opt_clf_rand))
print("R-Square Validation: ", r2_score(y_val, y_val_opt_clf_rand))
print("="*50)
print("MAE Train: ", mean_absolute_error(y_train, y_train_opt_clf_rand))
print("MAE Validation: ", mean_absolute_error(y_val, y_val_opt_clf_rand))
print("="*50)


# In[86]:


train["rand_param_grid"] = y_train_opt_clf_rand
valid["rand_param_grid"] = y_val_opt_clf_rand


# In[87]:


plt.figure(figsize=(20, 2))
sns.lineplot(data=train, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=train, x="date", y="rand_param_grid", marker='o', linestyle='')
plt.legend(["Train", "Predict"])

plt.figure(figsize=(20, 2))
sns.lineplot(data=valid, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=valid, x="date", y="rand_param_grid", marker='o', linestyle='')
plt.legend(["Valid", "Predict"])


# ### 3.5 **xgboost** + Validation (Timeseriessplit) + more param 

# In[88]:


pip install xgboost


# In[89]:


from xgboost import XGBRegressor


# In[90]:


clf_xgb = XGBRegressor(random_state=2022)


# In[91]:


param_grid = {
    'max_depth': [3, 6, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'colsample_bytree': np.arange(0.4, 1.0),
    'n_estimators': [100, 500, 1000]
}


# In[92]:


grid = GridSearchCV(estimator=clf_xgb, 
                    param_grid=param_grid, 
                    scoring="neg_mean_squared_error", 
                    cv=cv_ls,
                    n_jobs=-1)

grid.fit(X_train_std, y_train)


# In[93]:


print(grid.best_score_)
print(grid.best_params_)


# In[94]:


opt_clf_xgb = XGBRegressor(max_depth=3, learning_rate=0.1, colsample_bytree=0.4, n_estimators=100)
opt_clf_xgb.fit(X_train_std, y_train)


# In[96]:


y_train_opt_clf_xgb = opt_clf_xgb.predict(X_train_std)
y_val_opt_clf_xgb = opt_clf_xgb.predict(X_val_std)


# In[97]:


print("R-Square Train: ", r2_score(y_train, y_train_opt_clf_xgb))
print("R-Square Validation: ", r2_score(y_val, y_val_opt_clf_xgb))
print("="*50)
print("MAE Train: ", mean_absolute_error(y_train, y_train_opt_clf_xgb))
print("MAE Validation: ", mean_absolute_error(y_val, y_val_opt_clf_xgb))
print("="*50)


# In[98]:


train["xbg_param_grid"] = y_train_opt_clf_xgb
valid["xbg_param_grid"] = y_val_opt_clf_xgb


# In[99]:


plt.figure(figsize=(20, 2))
sns.lineplot(data=train, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=train, x="date", y="xbg_param_grid", marker='o', linestyle='')
plt.legend(["Train", "Predict"])

plt.figure(figsize=(20, 2))
sns.lineplot(data=valid, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=valid, x="date", y="xbg_param_grid", marker='o', linestyle='')
plt.legend(["Valid", "Predict"])


# ## feature_importances

# In[100]:


feature_import = opt_clf_xgb.feature_importances_
feature_import = pd.Series(feature_import, features_columns).sort_values(ascending= True)


# In[101]:


feature_import


# In[102]:


feature_import.plot(kind= 'barh')


# # Predict on unseen data (test-set)

# ### 1. import test-set and manipulate as train-set

# In[103]:


test_path = 'Offtake_regression_input/offtake_regression_test.csv'
df_test = pd.read_csv(test_path)

df_test.head()


# In[104]:


print(df_train.shape ,df_test.shape)


# In[105]:


df_test


# In[106]:


df_test['date'] = pd.to_datetime(df_test['date'])
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['week_of_year'] = df_test['date'].dt.week


# In[107]:


df_test['pro'].unique()


# In[108]:


df_test = df_test[df_test['pro'].isin(['RSP', '2for', 'point', '2nd_haft','3for'])]


# In[109]:


dummies = df_test[['pro']]
dummies_col = dummies.columns


# In[110]:


dummies_col


# In[111]:


df_test = pd.get_dummies(df_test, columns = dummies_col)


# In[112]:


df_test = df_test[['date','year','month','week_of_year','sales_in','stores','pro_2for','pro_RSP','offtake']]


# In[113]:


df_test = df_test.rename(columns = {'pro_RSP' : 'RSP'})


# In[114]:


df_test


# In[115]:


features_columns


# In[116]:


df_test['pro_3for'] = 0


# In[117]:


df_test = df_test[['date','year','month','week_of_year','sales_in','stores','pro_2for','pro_3for','RSP','offtake']]


# In[118]:


X_test = df_test[features_columns].values
y_test = df_test["offtake"].tolist()


# In[119]:


X_test


# In[120]:


y_test


# In[121]:


X_test_norm = norm.transform(X_test)
X_test_std = stand.transform(X_test_norm)


# In[122]:


plt.figure(figsize=(15, 2));
sns.distplot(X_train_std);
sns.distplot(X_test_std);
plt.legend(["train", "test"])


# ### Model selected

# #### 1. Randonforest

# In[123]:


y_train_opt_clf_rand = opt_clf_rand.predict(X_train_std)
y_test_opt_clf_rand = opt_clf_rand.predict(X_test_std)


# In[124]:


print("R-Square Train: ", r2_score(y_train, y_train_opt_clf_rand))
print("R-Square Test: ", r2_score(y_test, y_test_opt_clf_rand))
print("="*50)
print("MAE Train: ", mean_absolute_error(y_train, y_train_opt_clf_rand))
print("MAE Test: ", mean_absolute_error(y_test, y_test_opt_clf_rand))
print("="*50)


# In[125]:


train["rand_param_grid"] = y_train_opt_clf_rand
df_test["rand_param_grid_test"] = y_test_opt_clf_rand


# In[126]:


plt.figure(figsize=(20, 2))
sns.lineplot(data=train, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=train, x="date", y="rand_param_grid", marker='o', linestyle='')
plt.legend(["Train", "Predict"])

plt.figure(figsize=(20, 2))
sns.lineplot(data=df_test, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=df_test, x="date", y="rand_param_grid_test", marker='o', linestyle='')
plt.legend(["test", "Predict"])


# In[127]:


df_test


# #### 2. Xgboost

# In[128]:


y_train_opt_clf_xgb = opt_clf_xgb.predict(X_train_std)
y_test_opt_clf_xgb = opt_clf_xgb.predict(X_test_std)


# In[129]:


print("R-Square Train: ", r2_score(y_train, y_train_opt_clf_xgb))
print("R-Square Test: ", r2_score(y_test, y_test_opt_clf_xgb))
print("="*50)
print("MAE Train: ", mean_absolute_error(y_train, y_train_opt_clf_xgb))
print("MAE Test: ", mean_absolute_error(y_test, y_test_opt_clf_xgb))
print("="*50)


# In[130]:


train["xbg_param_grid"] = y_train_opt_clf_xgb
df_test["xbg_param_grid_test"] = y_test_opt_clf_xgb


# In[131]:


plt.figure(figsize=(20, 2))
sns.lineplot(data=train, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=train, x="date", y="xbg_param_grid", marker='o', linestyle='')
plt.legend(["Train", "Predict"])

plt.figure(figsize=(20, 2))
sns.lineplot(data=df_test, x="date", y="offtake", marker='o', linestyle='')
sns.lineplot(data=df_test, x="date", y="xbg_param_grid_test", marker='o', linestyle='')
plt.legend(["test", "Predict"])


# In[132]:


df_test

