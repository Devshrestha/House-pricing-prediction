import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from Multi_LR import LinearRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot1(X,Y,l,col):
    plt.scatter(X,Y,c=col)
    plt.xlabel(l)
    plt.ylabel('Prices')
    plt.show()

path = "D:\\Machine Learning\\Practice\\Surpervised Learning\\linearRegression\\Impli\\USA_Housing.csv"

df = pd.read_csv(path)

inputs = np.array(df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
           'Avg. Area Number of Bedrooms', 'Area Population']])

scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)

targets  = np.array(df['Price'])

x_train,x_test,y_train,y_test = train_test_split(inputs,targets,test_size =0.3)

plot1(inputs[:,0],targets,'Income avg','r')
plot1(inputs[:,1],targets,'house age','g')
plot1(inputs[:,2],targets,'Rooms','y')
plot1(inputs[:,3],targets,'Bedrooms','orange')

model = LinearRegressor()

model.train(x_train,y_train)

pre = model.predict(x_test)

score = model.score(y_test)

print('Score:',score)


plt.plot(np.sort(x_test[:,0]),np.sort(pre))
plt.show()