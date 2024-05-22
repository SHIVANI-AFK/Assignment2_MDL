import torch

def create_tensor_of_val(dimensions, val):
    return torch.ones(dimensions) * val

def calculate_elementwise_product(A, B):
    return A * B

def calculate_matrix_product(X, W):
    return torch.matmul(X, W.T)

def calculate_matrix_prod_with_bias(X, W, b):
    return torch.matmul(X, W.T) + b

def calculate_activation(sum_total):
    return torch.heaviside(sum_total, torch.tensor(0.0))

def calculate_output(X, W, b):
    return calculate_activation(calculate_matrix_prod_with_bias(X, W, b))