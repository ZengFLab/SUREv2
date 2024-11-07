import torch
from torch.utils.data import Dataset

import numpy as np
import textwrap

class CustomDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        return x, idx
    
class CustomDataset2(Dataset):
    def __init__(self, X, U):
        self.X = X
        self.U = U

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.U is None:
            u = x
        else:
            u = self.U[idx]
        return x, u, idx
    

class CustomDataset3(Dataset):
    def __init__(self, X, U, Y):
        self.X = X
        self.U = U
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.U is None:
            u = x
        else:
            u = self.U[idx]
        if self.Y is None:
            y = x
        else:
            y = self.Y[idx]
        return x, u, y, idx
    

class CustomMultiOmicsDataset(Dataset):
    def __init__(self, X1, X2):
        self.X1 = X1
        self.X2 = X2

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        x1 = self.X1[idx]
        x2 = self.X2[idx]
        return x1, x2, idx
    
class CustomMultiOmicsDataset2(Dataset):
    def __init__(self, X1, X2, U):
        self.X1 = X1
        self.X2 = X2
        self.U = U

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        x1 = self.X1[idx]
        x2 = self.X2[idx]
        if self.U is None:
            u=x1
        else:
            u = self.U[idx]
        return x1, x2, u, idx
    
    
def tensor_to_numpy(tensor):
    """
    Check if the tensor is on a CUDA device. If yes, detach it, move it to CPU,
    and convert to a NumPy array. If not, just detach and convert to NumPy.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        np.ndarray: The resulting NumPy array.
    """
    # Check if the input is a tensor
    if not isinstance(tensor, torch.Tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        raise ValueError("Input must be a torch Tensor.")

    # Detach the tensor from the computation graph
    tensor = tensor.detach()
    
    # Check if the tensor is on CUDA
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert to NumPy
    numpy_array = tensor.numpy()
    return numpy_array

def move_to_device(data, device):
    """
    Checks if the input data is a tensor. If not, converts it to a tensor,
    checks if the tensor is on the specified device, and moves it if necessary.

    Args:
        data (any): The input data to check (can be a tensor, list, NumPy array, etc.).
        device (str or torch.device): The device to check against (e.g., 'cpu', 'cuda', 'cuda:0').

    Returns:
        torch.Tensor: The tensor on the specified device.
    """
    # Convert input data to tensor if it's not already a tensor
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

    # Check if the device is a string, and convert it to torch.device if necessary
    device = torch.device(device) if isinstance(device, str) else device

    # Move the tensor to the specified device if necessary
    if data.device != device:
        data = data.to(device)
    
    return data


def convert_to_tensor(input_array, dtype=torch.float32, device=None):
    """
    Check if the input array is a torch tensor and convert it to a tensor if it is not.
    If dtype is specified, convert the tensor to the specified dtype if necessary.
    
    Parameters:
    - input_array: The input array to check and convert.
    - dtype: The desired data type for the resulting tensor (optional).
    
    Returns:
    - A torch tensor.
    """
    # Check if the input is already a torch tensor
    if isinstance(input_array, torch.Tensor):
        #print("Input is already a torch tensor.")
        # If dtype is specified, check and convert if necessary
        if dtype is not None and input_array.dtype != dtype:
            #print(f"Changing tensor dtype from {input_array.dtype} to {dtype}.")
            input_array = input_array.to(dtype)
        if device:
            input_array = move_to_device(input_array, device)
        return input_array  # Return the tensor unchanged if dtype matches

    else:
        # Convert to torch tensor
        #print("Input is not a torch tensor. Converting to torch tensor.")
        tensor = torch.tensor(input_array, dtype=dtype)
        if device:
            tensor = move_to_device(tensor, device)
        return tensor
    

class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

def pretty_print(long_text, width=120, color='green'):
    # Convert multiple spaces to a single space
    formatted_text = ' '.join(long_text.split())

    # Wrap the text to a specified width
    wrapped_text = textwrap.fill(formatted_text, width=width)

    # Define the indent for subsequent lines
    indent = '    '  # Four spaces for indentation

    # Split the wrapped text into lines
    lines = wrapped_text.split('\n')

    text_color = Colors.RESET
    if color.lower() == 'green':
        text_color = Colors.GREEN
    elif color.lower == 'yellow':
        text_color = Colors.YELLOW

    # Print the first line without indent
    print(text_color + lines[0] + Colors.RESET)

    # Print the subsequent lines with indent
    for line in lines[1:]:
        print(indent + text_color + line + Colors.RESET)

def find_partitions_greedy(numbers, num_groups):
    # Step 1: Calculate the target sum per group
    total_sum = sum(numbers)
    target_per_group = total_sum / num_groups

    # Initialize data structures
    groups = [[] for _ in range(num_groups)]  # Groups of numbers
    sums = [0] * num_groups  # Sums of each group
    indices = [[] for _ in range(num_groups)]  # Indices of numbers in original list

    # Step 2: Sort numbers and their indices based on value
    sorted_numbers_with_indices = sorted(enumerate(numbers), key=lambda x: -x[1])

    # Step 3: Distribute numbers to approach the target sum per group as close as possible
    for index, number in sorted_numbers_with_indices:
        # Find the group with the minimum sum
        min_group_index = sums.index(min(sums))
        groups[min_group_index].append(number)
        indices[min_group_index].append(index)
        sums[min_group_index] += number

    # Return the groups with their original indices
    return [(group, index_group) for group, index_group in zip(groups, indices)]