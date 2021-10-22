import numpy as np

input = np.array([0,0])
w0 = np.array([[0.4,0.2],[-0.6,0.7]])
w1 = np.array([0.1,-0.5])
b1 = np.array([0.3,0.1])
b2 = np.array([-0.4])

a1 = np.dot(input,w0) + b1
a2 = np.dot(a1,w1) + b2


'''layer_z = []
layer_a = []
a = state
layer_a.append(a)
for i in range(len(w_arrays)):
    z = (np.dot(w_arrays[i], a)) + biases[i]
    #a = sigmoid(z)
    a = s * 2 * ((1 / (1 + np.exp(-f * z))) - 0.5)
    layer_z.append(z)
    layer_a.append(a)'''