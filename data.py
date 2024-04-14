import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('Data\house-prices.csv')

df = df.drop(['ADDRESS','LONGITUDE','LATITUDE', 'POSTED_BY'], axis=1)


value_mapping = {'BHK': 1, 'RK': 0,}
df['BHK_OR_RK'] = df['BHK_OR_RK'].map(value_mapping)

print(df.head())


# data = df[df['TARGET(PRICE_IN_LACS)'] < 2000]
# data.reset_index(drop=True, inplace=True)

data = df

X = data.drop('TARGET(PRICE_IN_LACS)', axis=1)
y = data["TARGET(PRICE_IN_LACS)"]

print(X.head())
print(y.head())


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)


print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

print(X_test)