import numpy as np

class FC:
    def __init__(self, input_size : int, output_size : int, name : str, initialize_method : str="random"):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.initialize_method = initialize_method
        self.parameters = [self.initialize_weights(), self.initialize_bias()]
        self.input_shape = None
        self.reshaped_shape = None
    
    def initialize_weights(self):
        if self.initialize_method == "random":
            # TODO: Initialize weights with random values using np.random.randn
            return (np.random.randn(self.output_size, self.input_size) * 0.01)

        elif self.initialize_method == "xavier":
            lower, upper = -(1.0 / sqrt(self.input_size)), (1.0 / sqrt(self.input_size))
            random = np.random.randn(self.output_size, self.input_size)
            return lower + numbers * (upper - lower)

        elif self.initialize_method == "he":
            std = sqrt(2.0 / self.input_size)
            random = np.random.randn(self.output_size, self.input_size)
            return random * std

        else:
            raise ValueError("Invalid initialization method")
    
    def initialize_bias(self):
        # TODO: Initialize bias with zeros
        return np.zeros((self.output_size, 1))
    
    def forward(self, A_prev):
        """
        Forward pass for fully connected layer.
            args:
                A_prev: activations from previous layer (or input data)
                A_prev.shape = (batch_size, input_size)
            returns:
                Z: output of the fully connected layer
        """
        # NOTICE: BATCH_SIZE is the first dimension of A_prev
        self.input_shape = A_prev.shape
        A_prev_tmp = np.copy(A_prev)

        # TODO: Implement forward pass for fully connected layer
        if A_prev.ndim == 4: # check if A_prev is output of convolutional layer
            batch_size = A_prev.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T
        self.reshaped_shape = A_prev_tmp.shape
        
        # TODO: Forward part
        W, b = self.parameters[0], self.parameters[1]
        Z = []
        Z = W @ A_prev_tmp + b
        return Z
    
    def backward(self, dZ, A_prev):
        """
        Backward pass for fully connected layer.
            args:
                dZ: derivative of the cost with respect to the output of the current layer
                A_prev: activations from previous layer (or input data)
            returns:
                dA_prev: derivative of the cost with respect to the activation of the previous layer
                grads: list of gradients for the weights and bias
        """
        A_prev_tmp = np.copy(A_prev)
        batch_size = A_prev.shape[1]
        if A_prev.ndim == 4: # check if A_prev is output of convolutional layer
            batch_size = A_prev.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T

        # TODO: backward part
        W, b = self.parameters[0], self.parameters[1]
        dW = dZ @ A_prev_tmp.T / batch_size
        db = np.sum(dZ, axis=1, keepdims=True) / batch_size
        dA_prev = W.T @ dZ
        grads = [dW, db]
        # reshape dA_prev to the shape of A_prev
        if A_prev.ndim == 4:    # check if A_prev is output of convolutional layer
            dA_prev = dA_prev.T.reshape(A_prev.shape)
        return dA_prev, grads
    
    def update_parameters(self, optimizer, grads):
        """
        Update the parameters of the layer.
            args:
                optimizer: optimizer object
                grads: list of gradients for the weights and bias
        """
        self.parameters = optimizer.update(grads, self.name)
