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

            dw=(1/n) *np.dot(x.T,(y_pred-y))
            db=(1/n) *np.sum(y_pred-y)

            self.weights-=self.learning_rate*dw
            self.bias-=self.learning_rate*db
    
    def predict(self,x):
        return np.dot(x,self.weights)+self.bias
    @staticmethod
    def mse(y_pred,y):
        return np.mean((y_pred-y)**2)
    
X=np.array([
    [1],[2],[3]
])
Y=np.array([1,2,3])

for lr in [0.1,0.2,0.01,0.3,0.15]:
    model=LinearRegression(lr,500)
    model.fit_train(X,Y)
    y_pred=model.predict(X)
    print(f"Learning Rate: {lr} Prediction : {y_pred} MSE:{model.mse(y_pred,Y)}")
        
    

    
