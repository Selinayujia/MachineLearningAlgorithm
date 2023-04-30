import numpy as np
from tqdm import tqdm

class Linear_Regression():
    def __init__(self, alpha = 1e-3 , num_iter = 10000, early_stop = 100,
        intercept = True, init_weight = None, penalty = None, 
        lam = 1e-3, normalize = False, adaptive=True):
        """
        Linear Regression with gradient descent method. 

        Attributes:
        -------------
        alpha: Learning rate.
        num_iter: Number of iterations    
        early_stop: Number of steps without improvements 
                    that triggers a stop training signal
        intercept: True = with intercept (bias), False otherwise
        init_weight: Optional. The initial weights passed into the model, 
                                for debugging
        penalty: {None,l1,l2}. Define regularization type for regression. 
                  None: Linear Regression
                  l1: Lasso Regression
                  l2: Ridge Regression
        lam: regularization constant.
        normalize: True = normalize data, False otherwise
        adaptive: True = adaptive learning rate, False = fixed learning rate
        """
        self.alpha = alpha
        self.num_iter = num_iter
        self.early_stop = early_stop
        self.intercept = intercept
        self.init_weight = init_weight
        self.penalty = penalty
        self.lam = lam
        self.normalize = normalize
        self.adaptive = adaptive
    
    def fit(self, X, y):
        # initialize X, y
        self.X = X
        self.y = np.array([y]).T
        
        self.max = np.zeros((1, X.shape[1]))
        self.min = np.zeros((1, X.shape[1]))
        
        for i in range(self.X.shape[1]):
            self.max[:, i] = self.X[:, i].max() #[[996]]  column expression
            self.min[:, i] = self.X[:, i].min() #[[1]]
        

        ############### START TODO 1 ###############
        # Normalize the data using the formula provided in lecture
        if self.normalize:
            self.X = (self.X - self.min)/(self.max - self.min)
 
            
        ############### END TODO 1 ###############

        ############### START TODO 2 ###############
        # Add bias (if necessary) by concatanating a constant column into X
        # Hint: go through HW1 Q5 might be helpful
        if self.intercept:
            intercept_col = np.ones((self.X.shape[0], 1))
            self.X = np.concatenate((self.X, intercept_col), axis=1)
            
        ############### END TODO 2 ###############

        # initialize coefficient
        self.coef = self.init_weight if self.init_weight is not None\
        else np.array([np.random.uniform(-1,1,self.X.shape[1])]).T
        

        # start training, self.loss is used to record losses over iterations
        self.loss = []
        self.gradient_descent()
        
    def gradient(self):
        coef = -2 / len(self.X)
   

        ############### START TODO 3 ###############
        # Find prediction and gradient
        # Hint: Find the model's prediction from the given inputs with the 
        #       coefficient, then calculate the gradient
        # If you forgot the formula, find them in lecture 4 and 5
        
        #print('predicted',(np.matmul(self.X, self.coef)-self.y))
       # print('mul',np.dot((np.matmul(self.X, self.coef)-self.y),self.X))
        grad = coef * np.matmul(self.X.T, (np.matmul(self.X, self.coef)-self.y))
        
        grad = grad/max(grad)
        

        ############### END TODO 3 ###############

        ############### START TODO 4 ###############
        # Implement regularization penalty
        # Hint: Use self.lam
        if self.penalty == 'l2':
            grad += 2 * self.lam * self.coef
        elif self.penalty == 'l1':
            grad += self.lam * np.sign(self.coef)
        else: 
            pass
        ############### END TODO 4 ###############

        return grad
        
    def gradient_descent(self):
        print('Start Training')
        for i in range(self.num_iter):
            previous_y_hat = np.matmul(self.X,self.coef)
            print(previous_y_hat-self.y)
            grad = self.gradient()
            temp_coef = self.coef + (self.alpha * grad)
            
           # print('prev',self.coef, 'curr',temp_coef)
            #print('------------                     ')

            
            if self.penalty == 'l2':
                previous_reg_cost = 2 * self.lam * self.coef
                current_reg_cost = 2 * self.lam * temp_coef
            elif self.penalty == 'l1':
                previous_reg_cost = self.lam * np.sign(self.coef)
                current_reg_cost = self.lam * np.sign(temp_coef)
            else:
                previous_reg_cost = 0
                current_reg_cost = 0
            
            
  

            pre_error = np.sum(np.square(self.y - previous_y_hat)) + previous_reg_cost
            current_error = np.sum(np.square(self.y - np.matmul(self.X, temp_coef))) + current_reg_cost
            
            self.coef = temp_coef
            
            '''


            if current_error < pre_error:
                self.alpha *= 1.3  if self.adaptive else self.alpha
                self.coef = self.coef - (self.alpha * grad)
            else:
                self.alpha *= 0.9 if self.adaptive else self.alpha
                self.coef = temp_coef
                
            '''
                
            self.loss.append(float(current_error))
                        
            if len(self.loss) > self.early_stop and \
            self.loss[-1] >= max(self.loss[-self.early_stop:]):
                print('-----------------------')
                print(f'End Training (Early Stopped at iteration {i})')
                print(self.coef)
                return self
                
           
            if i % 1000000 == 0:
                print('-----------------------')
                print('Iteration: ' +  str(i))
                print('Coef: '+ str(self.coef))
                print('Loss: ' + str(current_error))
        print('-----------------------')            
        print('End Training')
        return self
    
    def predict(self, X):
        X_norm = np.zeros(X.shape)
        for i in range(X.shape[1]):
            X_norm[:, i] = (X[:, i] - self.min[:, i]) / (self.max[:, i] - self.min[:, i])
        X = X_norm
        ############### START TODO 9 ###############
        # add bias (if necessary, same as TODO 2)
        if self.intercept:

            intercept_col = np.ones((self.X.shape[0], 1))
            X = np.concatenate((X, intercept_col), axis=1)
        
            

        # Find the model's predictions
        # Hint: Use matrix multiplication ('@' might come in handy here)
        y = np.matmul(X,self.coef)
        return y
        ############### END TODO 9 ###############

    # Congrats! You have reached the end of this model's implementation :)
