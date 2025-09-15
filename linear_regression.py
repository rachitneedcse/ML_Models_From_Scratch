import numpy as np
class LinearRegression:
    def __init__(self,learning_rate,epochs):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.weights=None
        self.bias=None

    def fit_train(self,x,y):
        n,m=x.shape
        self.weights=np.zeros(m)
        self.bias=0

        for _ in range(self.epochs):

            y_pred=np.dot(x,self.weights)+self.bias

            dw=(1/n) *np.dot((y_pred-y),x.T)
            db=(1/n) *np.sum(y_pred-y)

            self.weights-=dw
            self.bias-=db
    
        
    

    
