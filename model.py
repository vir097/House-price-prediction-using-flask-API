import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle


df = pd.read_csv('Housing.csv')

dataset = df[['area','bedrooms','bathrooms','stories','parking','price']]

Y= dataset['price']
X= dataset.drop('price', axis=1)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0 )

regressor = LinearRegression()
regressor.fit(x_train, y_train)


pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

# print(model.predict([[8760, 4, 3, 4, 3]]))