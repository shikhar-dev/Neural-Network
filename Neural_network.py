import numpy as np

#DESCRIPTION
#There are three features x1,x2,x3 , when x3 is 1 output is 1 else 0
#So basically value of x1 and x2 dont matter
#We will start with random weights
# 1 Layer Neural Network
# Taking sigmoid as activation function
# deriv = True when we want the derivative of the function
# input Layer      output
#   ->o            layer
#   ->o
#   ->o             o  --- > Y'
#   ->o
def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input Data set
X = np.loadtxt("data.txt").T   # read input file

# ground label Truth or output

y = np.loadtxt("ans.txt")    #read ans

# seeding random numbers

np.random.seed(1)

# weight matrix
w1 = 2 * np.random.random((1, 3)) - 1
b1 = 2 * np.random.random((1, 1)) - 1

# Learning Rate
alpha = 0.1

for iter in xrange(10000):
    # forward propogation
    z1 = np.dot(w1, X) + b1  # z1(1,4)
    a1 = sigmoid(z1)            #predicated value

    m=np.shape(X)[1]      #calc no of test cases automatically , here dimension of X is returned 
    # backward propogation
    print 'ERROR : ', np.sum(y-a1)/m
    dz1 = a1 - y  # derivative
    dw1 = np.dot(dz1, X.T)/m                # For w1
    db1 = np.sum(dz1,axis=1,keepdims=True)/m     #For b1

    w1 = w1 - alpha * dw1    #Gradient Descent
    b1 = b1 - alpha * db1

print w1
print b1
#predict
x = np.array([[1],[0],[0]])
z1 = np.dot(w1, x) + b1  # z1(1,4)
a1 = sigmoid(z1)
print
print 'Predicted Value : ', a1[0][0] 


# Output is checked :)
#We can see weight corresponding to x3 is huge and rest is small :)
