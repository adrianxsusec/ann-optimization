"""
Comparing streaming to CUDA vs memory preallocation
"""


from line_profiler import profile
import torch
import numpy as np
np.random.seed(42)
X = np.random.random((2700, 1024))
y = np.random.random((2700, 1))
theta = np.random.random((8227,))
if torch.cuda.is_available():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

@profile
def pipeline_torch(X, y, theta):
    # Combine transfers using CUDA streams
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        X_torch = torch.from_numpy(X).cuda(device, non_blocking=True)
        y_torch = torch.from_numpy(y).cuda(device=device, non_blocking=True)
        theta_torch = torch.tensor(theta, dtype=torch.float64,requires_grad=True).cuda(device=device, non_blocking=True)
        torch.cuda.empty_cache()
        return X_torch, y_torch, theta_torch

@profile
def prealloc_torch(X, y, theta):
    # Pre-allocate GPU memory
    X_torch = torch.empty(X.shape, dtype=torch.float64, device=device)
    y_torch = torch.empty(y.shape, dtype=torch.float64, device=device)

    # Copy data directly to pre-allocated tensors
    X_torch.copy_(torch.from_numpy(X).cuda(device=device))
    y_torch.copy_(torch.from_numpy(y).cuda(device=device))

    theta_torch = torch.tensor(theta, dtype=torch.float64,
                             requires_grad=True).cuda(device=device)
    torch.cuda.empty_cache()
    return X_torch, y_torch, theta_torch

if __name__ == "__main__":
    X_torch, y_torch, theta_torch = pipeline_torch(X, y, theta)
    XX_torch , yy_torch, ttheta_torch = prealloc_torch(X, y, theta)

