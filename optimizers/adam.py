import numpy as np
# from .gradientdescent import GradientDescent

# TODO: Implement Adam optimizer
class Adam:
    def __init__(self, layers_list, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers_list
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.V = {}
        self.S = {}
        self.epoch = 1
        for l in layers_list:
            # TODO: Initialize V and S for each layer (v and s are lists of zeros with the same shape as the parameters)
            v = [0 for p in layers_list[l].parameters]
            s = [0 for p in layers_list[l].parameters]
            self.V[l] = v
            self.S[l] = s
        
    def update(self, grads, name, epoch=1):
        layer = self.layers[name]
        params = []
        # TODO: Implement Adam update
        for g in range(len(grads)):
            self.V[name][g] = self.beta1 * self.V[name][g] + (1 - self.beta1) * grads[g]
            self.S[name][g] = self.beta2 * self.S[name][g]  +(1 - self.beta2) * np.power(grads[g], 2)
            self.V[name][g] /= (1 - np.power(self.beta1, self.epoch)) # TODO: correct V
            self.S[name][g] /= (1 - np.power(self.beta1, self.epoch)) # TODO: correct S
            params.append(layer.parameters[g] - self.learning_rate * (self.V[name][g] / (np.sqrt(self.S[name][g]) + self.epsilon)))
        self.epoch += 1
        return params
