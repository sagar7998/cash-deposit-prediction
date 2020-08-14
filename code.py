# --------------
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv(path)
df.head()
print(df.columns)
df.columns = [i.lower().replace(' ','_') for i in df.columns]
print(df.columns)
df.fillna(np.nan)



# --------------
from sklearn.model_selection import train_test_split
df.set_index(keys='serial_number',inplace=True,drop=True)


# Code starts
df[['established_date','acquired_date']] = df[['established_date','acquired_date']].apply(pd.to_datetime)
#print(df[['established_date','acquired_date']])
X = df.drop('2016_deposits', axis=1)
y = df['2016_deposits']
print(X.head())
print(y.head())
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state = 3) 

# Code ends here


# --------------
# time_col = X_train.select_dtypes(exclude=[np.number,'O']).columns
time_col = ['established_date', 'acquired_date']

# Code starts here
for dataframe in [X_train, X_val]:
    for col_name in time_col:
        new_col_name = "since_"+col_name
        dataframe[new_col_name] = pd.datetime.now() - dataframe[col_name]
        print(dataframe[col_name].head())
        dataframe[new_col_name] = dataframe[new_col_name].apply(lambda x: float(x.days)/365)
        dataframe.drop(col_name, axis=1, inplace=True)



# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
cat = X_train.select_dtypes(include='O').columns.tolist()

# Code starts here
X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)

le = LabelEncoder()
for df in [X_train, X_val]:
    for col in cat:
        df[col] = le.fit_transform(df[col])

X_train_temp = pd.get_dummies(data = X_train, columns = cat)
X_val_temp = pd.get_dummies(data = X_val, columns = cat)

print(X_train_temp.head())


# Code ends here


# --------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Code starts here
dt = DecisionTreeRegressor(random_state = 5)
dt.fit(X_train, y_train)

accuracy = dt.score(X_val, y_val)
print(accuracy)
y_pred = dt.predict(X_val)
#print(y_pred)
mse = mean_squared_error(y_pred, y_val)
print("mse for DT:", mse)
rmse = np.sqrt(mean_squared_error(y_pred, y_val))

print("rmse for DecisionTreeRegressor: ",rmse)




# --------------
from xgboost import XGBRegressor


# Code starts here
xgb = XGBRegressor(max_depth=50, learning_rate=0.83, n_estimators=100)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_val)
print(y_pred)

accuracy = xgb.score(X_val, y_val)
print(accuracy)

mse = mean_squared_error(y_pred, y_val)
print(mse)

rmse = np.sqrt(mean_squared_error(y_pred, y_val))
print("rmse for XGBRegressor: ",rmse)



# Code ends here


