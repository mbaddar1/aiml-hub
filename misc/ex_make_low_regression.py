from sklearn.datasets import make_low_rank_matrix

if __name__ == "__main__":
    X = make_low_rank_matrix(n_samples=1000,n_features=100,effective_rank=10,tail_strength=0.0,random_state=0)
    pass