import torch
from garage.torch import global_device

def np_to_torch(array):
    """Numpy arrays to PyTorch tensors.
    Args:
        array (np.ndarray): Data in numpy array.
    Returns:
        torch.Tensor: float tensor on the global device.
    """
    return torch.from_numpy(array).float().to(global_device())
