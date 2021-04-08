


import numpy as np 
import matplotlib.pyplot as plt


def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])


epochs = 11000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,4,1


hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias =np.random.uniform(size=(1,hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons)) 
output_bias = np.random.uniform(size=(1,outputLayerNeurons))



#list of variables which is used for tracking errors in all four input cases
loss1=[]# input is [0,0]
loss2=[]# input is [0,1]
loss3=[]# input is [1,0]
loss4=[]# input is [1,1]


for i in range(epochs):
    #forward pass
	hidden_layer_output = sigmoid(np.dot(inputs,hidden_weights)+hidden_bias)
	predicted_output = sigmoid(np.dot(hidden_layer_output,output_weights)+output_bias)
	#backward_pass
	error = expected_output - predicted_output
	losses=0.5*(expected_output - predicted_output)**2   #cost function
	d_output = error * sigmoid_derivative(predicted_output) #derivative of output layer
	error_hidden_layer = np.dot(d_output,output_weights.T)
	d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output) #derivative of hidden layer
	output_weights += np.dot(hidden_layer_output.T,d_output) * lr #Updating output layer weights
	output_bias += np.sum(d_output,axis=0,keepdims=True) * lr #Updating output layer biases
	hidden_weights +=np.dot(inputs.T,d_hidden_layer) * lr #Updating hidden layer weights
	hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr #Updating hidden layer biases
	loss1.append(losses[0][0])
	loss2.append(losses[1][0])
	loss3.append(losses[2][0])
	loss4.append(losses[3][0])


print(predicted_output)
x=np.arange(1,epochs+1,1)
#Plotting graph of Losses versus Epochs
plt.plot(x,loss1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('Loss1.jpeg')
plt.show()
plt.plot(x,loss2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('Loss2.jpeg')
plt.show()
plt.plot(x,loss3)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('Loss3.jpeg')
plt.show()
plt.plot(x,loss4)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('Loss4.jpeg')
plt.show()
