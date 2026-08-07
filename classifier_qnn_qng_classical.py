import pennylane as qml
import torch
from pennylane import numpy as pnp
import numpy as np

"""
My Generic QNN which has 2 qubits and 3 layers. Should be trainable using any method.

Torch interface is used so weights and inputs are tensors.
"""


NUM_QUBITS = 2
NUM_LAYERS = 4 # originally 3
MAX_ITERATIONS = 400
BATCH_SIZE = 60

dev = qml.device("default.qubit", wires=NUM_QUBITS)

def stochastic_batch(x_train, y_train, size=BATCH_SIZE):
    # Returns a random sample of examples from the provided training set
    # Should hopefully work on tensors and numpy arrays
    indices = []
    index = np.random.randint(0, x_train.shape[0])
    
    for i in range(size):
        while index in indices:
            index = np.random.randint(0, x_train.shape[0])
        indices.append(index)
    
    batch_x = x_train[indices, :]
    batch_y = y_train[indices, :]
    
    return batch_x, batch_y

@qml.qnode(dev, interface='torch')
def quantum_circuit(weights, inputs=None):
    qml.AngleEmbedding(features=inputs, wires=[0, 1])
    
    qml.StronglyEntanglingLayers(weights, wires=[0, 1])
    
    return qml.expval(qml.PauliZ(0))

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def cross_entropy(w, input, label):
    pred = quantum_circuit(input, w)
    
    probs = sigmoid(pred)
    
    anti_probs = 1.0 - probs
    labels = label
    anti_labels = 1.0 - labels
    
    return -(labels * torch.log(probs) + anti_labels * torch.log(anti_probs))

weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=NUM_LAYERS, n_wires=NUM_QUBITS)
weight_shapes = {'weights': weight_shape}
torch.manual_seed(0)
qlayer_2 = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
