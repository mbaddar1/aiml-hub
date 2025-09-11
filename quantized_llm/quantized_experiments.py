"""
A Survey of Quantization Methods for Efficient Neural Network Inference
https://arxiv.org/pdf/2103.13630

AI Model Efficiency Toolkit (AIMET)
https://github.com/quic/aimet
"""
import numpy as np
import torch.optim
from torch import nn
from sklearn.datasets import make_regression
from loguru import logger
import matplotlib.pyplot as plt

class SimpleNNRegressor:
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.model = nn.Sequential(self.fc1, self.activation, self.fc2)

    def forward(self, x):
        return self.model(x)


def quantize1(A: torch.Tensor, S: float, Z: float,quant_dtype : torch.dtype):
    return torch.round(A / S).to(quant_dtype) - Z


def dequantize1(A_quant: torch.Tensor,S:float,Z:float):
    return S*(A_quant+Z)


if __name__ == "__main__":
    X, Y, A = make_regression(n_samples=10_000, n_targets=2, n_features=10, coef=True)
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    A = torch.tensor(A)
    y_hat = X @ A
    mse_baseline = torch.nn.MSELoss()(y_hat, Y).item()
    logger.info(f"MSE_baseline: {mse_baseline}")
    quant_dtype = torch.int8

    # piece of code to detect memory compression ratio

    S = 2
    Z = 0
    scales = list(range(1, 20))
    dequant_errors = []
    prediction_errors = []
    A_quant = quantize1(A, S, Z, quant_dtype)
    quant_size = A_quant.numel() * A_quant.element_size()
    orig_size = A.numel() * A.element_size()
    logger.info(f"Compression ratio: {(quant_size-orig_size)/orig_size*100}%")
    logger.info(f"Calculating quantization errors")
    for S in scales:
        A_quant = quantize1(A,S,Z,quant_dtype)
        A_dequantized = dequantize1(A_quant,S,Z)
        dequant_err = torch.linalg.norm(A_dequantized-A)
        #logger.info(f"Dequantization MSE @S={S}: {dequant_err}")
        y_hat2 = X @ A_dequantized.to(X.dtype)
        prediction_error = torch.nn.MSELoss()(y_hat2, Y).item()
        #logger.info(f"Prediction-MSE using Quantized MSE @ S = {S}: {prediction_error}")
        #logger.info(f"===")
        dequant_errors.append(dequant_err/torch.linalg.norm(A))
        prediction_errors.append(prediction_error / torch.linalg.norm(Y))
    logger.info("Plotting quantization errors")
    # Fit quadratic polynomials (degree=2) – you can increase degree if needed
    # Fit polynomial curves
    x = np.array(scales)
    recon_err = np.array(dequant_errors)
    pred_err = np.array(prediction_errors)
    dequant_fit = np.poly1d(np.polyfit(x, recon_err, 2))
    pred_fit = np.poly1d(np.polyfit(x, pred_err, 2))

    x_fit = np.linspace(min(x), max(x), 200)
    # Plot reconstruction error
    # plt.subplot(1, 2, 1)
    plt.plot(scales, dequant_errors, marker="o",label="Data")
    plt.plot(x_fit, dequant_fit(x_fit), "-", label="Fit")
    plt.xlabel("Scale (S)")
    plt.ylabel("Reconstruction Error (||A_dequant - A||)")
    plt.title("Reconstruction Error vs Scale")
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("reconstruction_error.png")
    plt.clf()

    # Plot prediction MSE
    # plt.subplot(1, 2, 2)
    plt.plot(scales, prediction_errors, marker="o",label="Data")
    plt.plot(x_fit, pred_fit(x_fit), "-", label="Fit")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Scale (S)")
    plt.ylabel("Prediction MSE")
    plt.title("Prediction MSE vs Scale")
    plt.savefig("prediction_error.png")
    plt.clf()



