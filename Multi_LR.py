import numpy as np 

class LinearRegressor:

    def train(self,features,targets,regularization_rate=0):
        self.X = features
        self.y = targets
        self.lamb= regularization_rate
        self.D = len(self.X[0]) # Dimesnsionality
        self.w = np.linalg.solve(self.lamb*np.eye(self.D)+self.X.T.dot(self.X),self.X.T.dot(self.y))

    def predict(self,features):

        self.x = features
        self.y_hat = self.x.dot(self.w)

        return self.y_hat

    def score(self,actual_values):

        #r^2 = 1-(ssr/sst)
        SSR = actual_values-self.y_hat
        SST = actual_values-self.y_hat.mean()

        r2  = 1 - SSR.dot(SSR)/SST.dot(SST)

        return r2