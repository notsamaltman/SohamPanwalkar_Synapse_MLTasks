# Exam Oracle 3000 - Forward Propagation Skeleton (1 Hidden Layer)

import numpy as np

# 1. Define sigmoid activation (In case you have tried different activation function code that here instead of sigmoid)
def sigmoid(z):
    return 1/(1+np.exp(-z))

def neuralnet(study_hours, attendance, quiz_score):
    # Input vector (3 features, 1 example)
    X = np.array([[study_hours], 
                [attendance], 
                [quiz_score]])   

    # Hidden layer (let’s say 3 neurons)
    W1 = np.array([[ 2, -5,  8], 
                [ 1,  3, -2], 
                [ 4, -1,  5]])  
    b1 = np.array([[1], 
                [4], 
                [-10]])

    # 4. Forward Propagation
    # Input → Hidden
    Z1 = np.dot(W1, X)+b1   # dot product + bias
    A1 = sigmoid(Z1)  # apply activation magic

    # Output layer (2 neurons: Pass, Fail)
    W2 = np.array([[1, -2, 3],
                [-1, 2, -3]])   # shape (2,3)
    b2 = np.array([[0],
                [1]])           # shape (2,1)

    Z2 = np.dot(W2, A1) + b2       # (2x3) · (3x1) + (2x1) = (2x1)
    A2 = sigmoid(Z2)

    # 5. Final Prediction
    print("Oracle prediction (Pass, Fail):", A2)

for i in range(3):
    neuralnet(10, np.random.randint(0, 10), np.random.randint(0, 10))


#TRY VERIFYING THE RESULTS WITH YOUR WEIGHTS AND BIASES THAT YOU HAVE TRIED ON PEN PAPER!!