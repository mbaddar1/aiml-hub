import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.linalg import svdvals
from sklearn.datasets import make_regression
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from flow_matching.sandbox.DMRG.spectral_rank import compute_spectral_entropy, compute_spectral_rank
def omega_approx(beta):
    """Return an approximate omega value for given beta. Equation (5) from Gavish 2014."""
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
def lambda_star(beta):
    """Return lambda star for given beta. Equation (11) from Gavish 2014."""
    return np.sqrt(2 * (beta + 1) + (8 * beta) /
                   (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1)))
def svht(X, sigma=None, sv=None):
    """Return the optimal singular value hard threshold (SVHT) value.
    `X` is any m-by-n matrix. `sigma` is the standard deviation of the
    noise, if known. Optionally supply the vector of singular values `sv`
    for the matrix (only necessary when `sigma` is unknown). If `sigma`
    is unknown and `sv` is not supplied, then the method automatically
    computes the singular values."""

    try:
        m, n = sorted(X.shape)  # ensures m <= n
    except:
        raise ValueError('invalid input matrix')
    beta = m / n  # ratio between 0 and 1
    if sigma is None:  # sigma unknown
        if sv is None:
            sv = svdvals(X)
        sv = np.squeeze(sv)
        if sv.ndim != 1:
            raise ValueError('vector of singular values must be 1-dimensional')
        return np.median(sv) * omega_approx(beta)
    else:  # sigma known
        return lambda_star(beta) * np.sqrt(n) * sigma
def compute_optimal_truncated_rank(A:torch.Tensor,eps:float=1e-3):
    U,S,Vh = torch.linalg.svd(A)
    thr = svht(X=A)
    S_np = S.detach().cpu().numpy()
    r = len(S_np[S_np>=thr])
    return r

def compute_stable_rank(A: torch.Tensor) -> float:
    """
    Compute stable rank of a matrix A using PyTorch.
    srank(A) = ||A||_F^2 / ||A||_2^2
    https://www.cs.ubc.ca/~nickhar/W12/Lecture15Notes.pdf
    """
    U,S,Vh = torch.linalg.svd(A, full_matrices=False)
    S_np = S.detach().cpu().numpy()
    srank = np.sum([S_np**2])/S_np[0]**2
    return srank

class SimpleNN(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, non_lin: torch.nn.Module,data_type:torch.dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.non_lin = non_lin
        if "fc1" in kwargs:
            assert isinstance(kwargs["fc1"], torch.nn.Linear)
            self.fc1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim,dtype=data_type)
            with torch.no_grad():
                self.fc1.weight = torch.clone(kwargs["fc1"].weight)
                self.fc1.bias = torch.clone(kwargs["fc1"].bias)
        else:
            self.fc1 = torch.nn.Linear(in_dim, hidden_dim,dtype=data_type)
        self.low_rank = False
        if "fc21" in kwargs.keys() and "fc22" in kwargs.keys():
            self.low_rank = True
            assert isinstance(kwargs["fc21"], torch.nn.Linear)
            assert isinstance(kwargs["fc22"], torch.nn.Linear)
            assert kwargs["fc21"].bias == False
            assert kwargs["fc22"].bias == False
            with torch.no_grad():
                self.fc21 = torch.nn.Linear(in_features=hidden_dim, out_features=kwargs["fc21"].weight.shape[1],dtype=data_type)
                self.fc21.weight = torch.clone(kwargs["fc21"].weight)
                self.fc22 = torch.nn.Linear(in_features=kwargs["fc22"].weight.shape[0], out_features=hidden_dim)
                self.fc22.weight = torch.clone(kwargs["fc22"].weight)
        else:
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False,dtype=data_type)

        if "fc3" in kwargs.keys():
            assert isinstance(kwargs["fc3"], torch.nn.Linear)
            with torch.no_grad():
                self.fc3.weight = torch.clone(kwargs["fc3"].weight)
                self.fc3.bias = torch.clone(kwargs["fc3"].bias)
        else:
            self.fc3 = torch.nn.Linear(hidden_dim, out_dim,dtype=data_type)

        if self.low_rank:
            self.model = torch.nn.Sequential(self.fc1, self.non_lin, self.fc21, self.fc22, self.non_lin, self.fc3)
        else:
            self.model = torch.nn.Sequential(self.fc1, self.non_lin, self.fc2, self.non_lin, self.fc3)

    def forward(self, X):
        return self.model(X)


if __name__ == "__main__":
    n = 10_000
    x_dim = 10
    y_dim = 4
    n_epochs = 1000
    nn_hidden_dim = 100
    non_lin = torch.nn.ReLU()
    X, Z = make_regression(n_samples=n, n_features=10, n_targets=y_dim)
    data_type = torch.float64
    X = torch.tensor(X,dtype=data_type)
    Z = torch.tensor(Z,dtype=data_type)

    Y = non_lin(Z)

    batch_size = 512
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    nn_model = SimpleNN(in_dim=x_dim, out_dim=y_dim, hidden_dim=nn_hidden_dim, non_lin=non_lin,data_type=data_type)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    alpha = 0.1
    smooth_loss = None
    learning_curve = []
    W_entropy=[]
    W_entropy_rank = []
    W_stable_ranks = []
    with torch.no_grad():
        preds = nn_model(X)
        loss = loss_fn(preds, Y)
        H = compute_spectral_entropy(nn_model.fc2.weight)
        spectral_rank = compute_spectral_rank(nn_model.fc2.weight)
        stable_rank_val = compute_stable_rank(nn_model.fc2.weight)
    with tqdm(iterable=range(n_epochs)) as pbar:
        for epoch in pbar:
            nn_model.train()
            epoch_loss = 0.0
            for xb, yb in dataloader:
                optimizer.zero_grad()
                preds = nn_model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            avg_loss = epoch_loss / len(dataset)
            smooth_loss = avg_loss if smooth_loss is None else alpha * avg_loss + (1 - alpha) * avg_loss
            pbar.set_postfix_str(f"MSE = {smooth_loss}")
            learning_curve.append(smooth_loss)
            with torch.no_grad():
                H = compute_spectral_entropy(nn_model.fc2.weight)
                spectral_rank = compute_spectral_rank(nn_model.fc2.weight)
                W_entropy.append(H)
                W_entropy_rank.append(spectral_rank)
                stable_rank_val = compute_stable_rank(nn_model.fc2.weight)
                W_stable_ranks.append(stable_rank_val)

    # compute svht rank at the end. Cannot during training as it is not robust to noise
    with torch.no_grad():
        svht_rank = compute_optimal_truncated_rank(A=nn_model.fc2.weight, eps=1e-3)
    plt.plot(learning_curve, marker="o")  # line plot with markers
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title("learning curve (log-scale)")
    plt.grid(True, which="both", ls="--")
    plt.savefig("learning_curve.png")
    #
    plt.clf()
    plt.plot(W_entropy, marker="o")  # line plot with markers
    plt.xlabel("epoch")
    plt.ylabel("entropy")
    plt.title("entropy curve")
    plt.grid(True)  # add grid
    plt.savefig("entropy_curve.png")
    #
    plt.clf()
    plt.plot(W_entropy_rank, marker="o")  # line plot with markers
    plt.xlabel("epoch")
    plt.ylabel("entropy rank")
    plt.title("entropy rank curve")
    plt.grid(True)  # add grid
    plt.savefig("entropy_rank_curve.png")
    #
    plt.clf()
    plt.plot(W_stable_ranks, marker="o")  # line plot with markers
    plt.xlabel("epoch")
    plt.ylabel("stable rank")
    plt.title("stable rank curve")
    plt.grid(True)  # add grid
    plt.savefig("stable_rank_curve.png")