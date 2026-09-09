"""
Deep Generative MNAR Imputer (not-MIWAE inspired).

Implements a deep latent variable model that jointly models the data distribution X
and the missingness mechanism mask M under Not-Missing-At-Random (MNAR) regimes.

Model Architecture:
  Latent Prior:   z ~ N(0, I_d)
  Data Decoder:   x | z ~ N(mu_theta(z), diag(sigma_theta^2(z)))
  Mask Mechanism: m_j | x_j ~ Bernoulli(sigmoid(w_j * x_j + b_j))  (Self-masking dependency)
  Encoder:        z | (x_obs, m) ~ N(mu_psi(x_obs, m), diag(sigma_psi^2(x_obs, m)))

References:
Ipsen, N. B., Mattei, P. A., & Frellsen, J. (2021).
How to Deal with Missing Not at Random Data: A Missingness-Agnostic Deep Generative Approach.
Advances in Neural Information Processing Systems (NeurIPS), 34, 19694-19707.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class DeepMNARAutoencoder(nn.Module):
        """Joint VAE modeling data X and missingness mask M."""

        def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 64):
            super().__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim

            # Encoder takes zero-filled observed values + mask
            self.encoder = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

            # Data Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

            # Self-masking MNAR mechanism: logit P(M_j = 1 | X_j) = w_j * X_j + b_j
            self.mask_weights = nn.Parameter(torch.zeros(input_dim))
            self.mask_biases = nn.Parameter(torch.zeros(input_dim))

        def encode(
            self, x_zero: torch.Tensor, mask: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            inp = torch.cat([x_zero, mask], dim=-1)
            h = self.encoder(inp)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h).clamp(-8.0, 4.0)
            return mu, logvar

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.decoder(z)
            return out

        def forward(
            self, x_zero: torch.Tensor, mask: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encode(x_zero, mask)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar


class DeepGenerativeMNARImputer(BaseEstimator, TransformerMixin):
    """
    Deep Generative MNAR Imputer based on joint data-mask variational autoencoders.

    Parameters
    ----------
    latent_dim : int, default=8
        Dimensionality of latent space z.
    hidden_dim : int, default=64
        Hidden layer width.
    epochs : int, default=60
        Number of training epochs.
    batch_size : int, default=64
        Batch size.
    lr : float, default=1e-3
        Learning rate.
    random_state : Optional[int], default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        random_state: Optional[int] = 42,
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state

        self.model_: Optional[Any] = None
        self.columns_: List[str] = []
        self.means_: np.ndarray = np.array([])
        self.stds_: np.ndarray = np.array([])
        self.feature_names_in_: Union[List[str], np.ndarray] = []
        self.n_features_in_: int = 0

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> "DeepGenerativeMNARImputer":
        """Train the deep generative MNAR model on observed data."""
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for DeepGenerativeMNARImputer. Install with `pip install torch`."
            )

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.n_features_in_ = len(self.feature_names_in_)
        else:
            self.n_features_in_ = int(X.shape[1])
            if hasattr(self, "feature_names_in_"):
                delattr(self, "feature_names_in_")

        df = self._to_dataframe(X)
        self.columns_ = list(df.columns)
        X_arr = df.to_numpy(dtype=float, copy=True)
        n, p = X_arr.shape

        # Mask: 1 if observed, 0 if missing
        mask_arr = (~np.isnan(X_arr)).astype(np.float32)

        # Standardize using observed stats
        self.means_ = np.nanmean(X_arr, axis=0)
        self.stds_ = np.nanstd(X_arr, axis=0)
        self.stds_ = np.where(self.stds_ < 1e-6, 1.0, self.stds_)

        X_norm = (X_arr - self.means_) / self.stds_
        X_zero = np.nan_to_num(X_norm, nan=0.0).astype(np.float32)

        # Prepare PyTorch Tensors
        tensor_x = torch.tensor(X_zero, dtype=torch.float32)
        tensor_m = torch.tensor(mask_arr, dtype=torch.float32)

        dataset = TensorDataset(tensor_x, tensor_m)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, n), shuffle=True)

        self.model_ = DeepMNARAutoencoder(
            input_dim=p,
            latent_dim=min(self.latent_dim, max(2, p // 2)),
            hidden_dim=self.hidden_dim,
        )
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_m in loader:
                optimizer.zero_grad()
                x_recon, mu, logvar = self.model_(batch_x, batch_m)

                # 1. Reconstruction loss on observed entries only
                recon_loss = ((x_recon - batch_x) ** 2 * batch_m).sum() / (batch_m.sum() + 1e-8)

                # 2. KL divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_x.size(0)

                # 3. MNAR mask likelihood: P(M | X_recon)
                # Self-masking logit
                mask_logits = x_recon * self.model_.mask_weights + self.model_.mask_biases
                bce_mask = nn.functional.binary_cross_entropy_with_logits(
                    mask_logits, batch_m, reduction="mean"
                )

                total_loss = recon_loss + 0.01 * kl_loss + 0.5 * bce_mask
                total_loss.backward()
                optimizer.step()

        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Impute missing values using the trained joint generative model."""
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for DeepGenerativeMNARImputer. Install with `pip install torch`."
            )
        check_is_fitted(self, "is_fitted_")
        assert self.model_ is not None
        n_features = (
            X.shape[1]
            if hasattr(X, "shape") and len(X.shape) > 1
            else len(getattr(X, "columns", []))
        )
        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input."
            )
        if (
            isinstance(X, pd.DataFrame)
            and hasattr(self, "feature_names_in_")
            and self.feature_names_in_ is not None
        ):
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError(
                    f"The feature names should match those that were passed during fit. "
                    f"Expected {list(self.feature_names_in_)}, got {list(X.columns)}"
                )

        df = self._to_dataframe(X).copy()
        X_arr = df.to_numpy(dtype=float, copy=True)
        mask_arr = (~np.isnan(X_arr)).astype(np.float32)

        X_norm = (X_arr - self.means_) / self.stds_
        X_zero = np.nan_to_num(X_norm, nan=0.0).astype(np.float32)

        tensor_x = torch.tensor(X_zero, dtype=torch.float32)
        tensor_m = torch.tensor(mask_arr, dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            mu_z, _ = self.model_.encode(tensor_x, tensor_m)
            x_recon_norm = self.model_.decode(mu_z).cpu().numpy()

        # Unstandardize
        x_recon = x_recon_norm * self.stds_ + self.means_

        # Replace only missing values
        mis_indices = np.where(mask_arr == 0)
        X_arr[mis_indices] = x_recon[mis_indices]

        return pd.DataFrame(X_arr, columns=self.columns_, index=df.index)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        if input_features is not None:
            if len(input_features) != self.n_features_in_:
                raise ValueError(
                    f"input_features should have length equal to number of features ({self.n_features_in_}), "
                    f"got {len(input_features)}"
                )
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_") and self.feature_names_in_ is not None:
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

    def _to_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if hasattr(self, "feature_names_in_") and self.feature_names_in_ is not None:
            cols = [str(c) for c in self.feature_names_in_]
        else:
            cols = [f"x{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)
