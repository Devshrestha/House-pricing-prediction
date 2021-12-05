import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

def plot1(X,Y,l,col):
    plt.scatter(X,Y,c=col)
    plt.xlabel(l)
    plt.ylabel('Prices')
    plt.show()

path = "D:\\Machine Learning\\Practice\\Surpervised Learning\\linearRegression\\Impli\\USA_Housing.csv"

df = pd.read_csv(path)

inputs = np.array(df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
           'Avg. Area Number of Bedrooms', 'Area Population']])

targets  = np.array(df['Price'])



#plot1(inputs[:,0],targets,'Income avg','r')
#plot1(inputs[:,1],targets,'house age','g')
#plot1(inputs[:,2],targets,'Rooms','y')
#plot1(inputs[:,4],targets,'Population','orange')

N,D = inputs.shape

w = np.random.randn(D)/np.sqrt(D)

alpha = 0.00001
l1 = 10
cost = []

for i in range(100):
    yhat = inputs.dot(w)
    delta = yhat - targets
    if i < 50:
        pass
    w = w - alpha*(inputs.T.dot(delta)+l1*np.sign(w))

    mse = 1-delta.dot(delta)/N
    cost.append(mse)


print(cost)