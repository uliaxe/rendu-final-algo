# neural_network.py
import numpy as np

class NeuralNetwork:
    def __init__(self, input_nodes=13, hidden_nodes=16, output_nodes=4, weights=None, biases=None):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        if weights is None:
            # He Initialization
            self.weights_input_hidden = np.random.randn(input_nodes, hidden_nodes) * np.sqrt(2 / input_nodes)
            self.weights_hidden_output = np.random.randn(hidden_nodes, output_nodes) * np.sqrt(2 / hidden_nodes)
            
            # Biases initialized to 0
            self.bias_hidden = np.zeros(hidden_nodes)
            self.bias_output = np.zeros(output_nodes)
        else:
            self.weights_input_hidden, self.weights_hidden_output = weights
            if biases is None:
                 # If loading old weights without biases, init zeros
                self.bias_hidden = np.zeros(hidden_nodes)
                self.bias_output = np.zeros(output_nodes)
            else:
                self.bias_hidden, self.bias_output = biases

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        # Stability fix: subtract max
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def forward(self, inputs):
        inputs = np.array(inputs)
        
        # Hidden layer
        # (Input x Weights) + Bias
        # Inputs: (13,) or (13,1)
        # Weights: (13, 16)
        # Result: (16,)
        
        hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden = self.relu(hidden)
        
        # Output layer
        output = np.dot(hidden, self.weights_hidden_output) + self.bias_output
        output = self.softmax(output)
        
        return output

    def get_weights(self):
        return [self.weights_input_hidden.copy(), self.weights_hidden_output.copy()]

    def get_biases(self):
        return [self.bias_hidden.copy(), self.bias_output.copy()]

    def mutate(self, rate=0.1):
        # Mutate weights
        for w in [self.weights_input_hidden, self.weights_hidden_output]:
            mask = np.random.random(w.shape) < rate
            w[mask] += np.random.randn(*w[mask].shape) * 0.5
            
        # Mutate biases
        for b in [self.bias_hidden, self.bias_output]:
            mask = np.random.random(b.shape) < rate
            b[mask] += np.random.randn(*b[mask].shape) * 0.5

    @staticmethod
    def crossover(p1, p2):
        w1 = p1.get_weights()
        w2 = p2.get_weights()
        b1 = p1.get_biases()
        b2 = p2.get_biases()
        
        child_w = []
        for a, b in zip(w1, w2):
            mask = np.random.rand(*a.shape) > 0.5
            child_w.append(np.where(mask, a, b))
            
        child_b = []
        for a, b in zip(b1, b2):
            mask = np.random.rand(*a.shape) > 0.5
            child_b.append(np.where(mask, a, b))
            
        return NeuralNetwork(weights=child_w, biases=child_b)
