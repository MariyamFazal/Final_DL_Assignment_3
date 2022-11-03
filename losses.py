import numpy as np

# loss function and its derivative
def Mean_Squared_Error(y_true, y_predicted):
    return np.mean(np.power(y_true-y_predicted, 2));

def Mean_Squared_Error_p(y_true, y_predicted):
    return 2*(y_predicted-y_true)/y_true.size;