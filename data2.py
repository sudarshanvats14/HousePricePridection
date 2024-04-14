import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('./Data/kc_house_data.csv')

print(data.head())

data.info()

data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].apply(lambda date:date.month)
data['year'] = data['date'].apply(lambda date:date.year)

# num_col = ['price','area']

# cat_col = ['bedrooms','bathrooms','stories','mainroad','guestroom','basement',
#            'hotwaterheating','airconditioning','parking','prefarea','furnishingstatus']data.isna().sum()


data.isna().sum()


data = data.drop('date',axis=1)
data = data.drop('id',axis=1)
data = data.drop('zipcode',axis=1)

data = data[data['price'] <= 1.00E+06]
data.reset_index(drop=True, inplace=True)


X = data.drop('price',axis =1).values
y = data['price'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(float))
X_test = s_scaler.transform(X_test.astype(float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



print(X_train.shape)
print(X_test.shape)

