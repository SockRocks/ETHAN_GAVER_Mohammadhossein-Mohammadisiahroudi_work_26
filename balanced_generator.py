import numpy as np
import torch
from sklearn.model_selection import train_test_split

def f(x, noise=True):
    # Noise creates a 25% chance the sign of the output will incorrectly flip
    real_val = 2 * x[0] + x[1] >= 6
    
    real_val = real_val if real_val else 0
    
    nse = np.random.randint(1,4) if noise else 4
        
    return real_val if nse > 1 else int(not real_val)


def generate_rand_vector():
    # Generates random vectors such that 50% will make f true
    return np.array([np.random.uniform(0,3), np.random.uniform(0,6)])


def generate_dataset(sample_size):
    """
    Generates a balanced dataset of the specified sample size. Returns x_train, y_train, x_test, y_test as
    pytorch 32-bit floating pt tensors
    """
    
    x_samples = []
    for x in range(sample_size):
        x_ds = torch.from_numpy(generate_rand_vector()).to(dtype=torch.float64)
        x_samples.append(x_ds)
        
    x_set = torch.vstack(x_samples)
    y_set = torch.tensor([[f(row, noise=True)] for row in x_set.unbind(0)])

    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, train_size=0.6, random_state=1)
    
    print(type(x_train))
    x_train = torch.tensor(x_train).to(dtype=torch.float32)
    y_train = torch.tensor(y_train).to(dtype=torch.float32)

    x_test = torch.tensor(x_test).to(dtype=torch.float32)
    y_test = torch.tensor(y_test).to(dtype=torch.float32)
    
    return x_train, y_train, x_test, y_test