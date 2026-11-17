"""Multitask Deep Learning (MLP) Architecture for Indoor Localization.

Implements:
1. `MultitaskMLP`: PyTorch neural network featuring a shared trunk and 4 task heads:
   - Head 1: Building classification (3 classes, CrossEntropyLoss)
   - Head 2: Floor classification (5 classes, CrossEntropyLoss)
   - Head 3: Longitude regression (Continuous coordinate, Smooth L1 / MSE loss)
   - Head 4: Latitude regression (Continuous coordinate, Smooth L1 / MSE loss)
2. `CoordinateStandardizer`: Zero-leakage target coordinate scaler fitted strictly on training data.
3. `MultitaskLoss`: Weighted loss aggregator combining classification and regression objectives.
4. `IndoorAIMultitaskMLP`: High-level scikit-learn compatible estimator with full training,
   early stopping, learning rate scheduling, metric tracking, and serialization.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.core.logging import setup_logger

logger = setup_logger("MultitaskMLP")

# Non-feature columns strictly prohibited from entering model input
FORBIDDEN_COLUMNS: Set[str] = {
    "BUILDINGID",
    "FLOOR",
    "BUILDING_INDEX",
    "FLOOR_INDEX",
    "BUILDING_FLOOR_INDEX",
    "LONGITUDE",
    "LATITUDE",
    "SPACEID",
    "RELATIVEPOSITION",
    "USERID",
    "PHONEID",
    "TIMESTAMP",
    "original_row_index",
    "sample_weight",
}

# Building-Floor combination index mapping (13 valid combinations across 3 buildings)
BUILDING_FLOOR_MAP: Dict[str, int] = {
    "0_0": 0, "0_1": 1, "0_2": 2, "0_3": 3,
    "1_0": 4, "1_1": 5, "1_2": 6, "1_3": 7,
    "2_0": 8, "2_1": 9, "2_2": 10, "2_3": 11, "2_4": 12,
}
INV_BUILDING_FLOOR_MAP: Dict[int, str] = {v: k for k, v in BUILDING_FLOOR_MAP.items()}


@dataclass
class MLPHyperparameters:
    """Hyperparameters for Multitask MLP architecture and training."""
    input_dim: int = 1058
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropouts: List[float] = field(default_factory=lambda: [0.30, 0.25])
    num_buildings: int = 3
    num_floors: int = 5
    lr: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 120
    patience: int = 15
    lr_factor: float = 0.5
    lr_patience: int = 5
    min_lr: float = 1e-6
    loss_type: str = "smooth_l1"  # "smooth_l1" or "mse"
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "building": 1.0,
            "floor": 2.0,
            "longitude": 1.0,
            "latitude": 1.0,
        }
    )
    scale_summary: bool = True
    summary_scaler_type: str = "minmax"  # "minmax" or "standard"
    scale_coordinates: bool = True
    coordinate_scaler_type: str = "standard"  # "standard" or "minmax"
    random_seed: int = 42
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CoordinateStandardizer:
    """Zero-leakage coordinate standardizer fitted strictly on training targets.
    
    Standardizes continuous Longitude and Latitude targets to zero mean and unit variance
    during neural network gradient descent, preventing coordinate loss from dominating
    cross-entropy classification loss.
    """

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler(feature_range=(-1.0, 1.0))
        self.is_fitted = False
        self.coord_cols = ["LONGITUDE", "LATITUDE"]

    def fit(self, coords: Union[pd.DataFrame, np.ndarray]) -> "CoordinateStandardizer":
        """Fit scaler strictly on training coordinates."""
        arr = coords.values if isinstance(coords, pd.DataFrame) else np.asarray(coords)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.scaler.fit(arr)
        self.is_fitted = True
        return self

    def transform(self, coords: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform coordinates to standardized space."""
        if not self.is_fitted:
            raise ValueError("CoordinateStandardizer is not fitted.")
        arr = coords.values if isinstance(coords, pd.DataFrame) else np.asarray(coords)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return self.scaler.transform(arr)

    def inverse_transform(self, coords_scaled: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Inverse transform standardized coordinates back to original coordinate units."""
        if not self.is_fitted:
            raise ValueError("CoordinateStandardizer is not fitted.")
        arr = coords_scaled.values if isinstance(coords_scaled, pd.DataFrame) else np.asarray(coords_scaled)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return self.scaler.inverse_transform(arr)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scaler parameters to JSON-serializable dict."""
        if not self.is_fitted:
            return {"is_fitted": False, "scaler_type": self.scaler_type}
        
        state: Dict[str, Any] = {
            "is_fitted": True,
            "scaler_type": self.scaler_type,
        }
        if isinstance(self.scaler, StandardScaler):
            state["mean"] = self.scaler.mean_.tolist()
            state["scale"] = self.scaler.scale_.tolist()
            state["var"] = self.scaler.var_.tolist()
        elif isinstance(self.scaler, MinMaxScaler):
            state["min"] = self.scaler.min_.tolist()
            state["scale"] = self.scaler.scale_.tolist()
            state["data_min"] = self.scaler.data_min_.tolist()
            state["data_max"] = self.scaler.data_max_.tolist()
        return state

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "CoordinateStandardizer":
        """Restore scaler from serialized dictionary."""
        scaler_type = state.get("scaler_type", "standard")
        inst = cls(scaler_type=scaler_type)
        if state.get("is_fitted", False):
            inst.is_fitted = True
            if scaler_type == "standard" and "mean" in state:
                inst.scaler.mean_ = np.array(state["mean"], dtype=np.float64)
                inst.scaler.scale_ = np.array(state["scale"], dtype=np.float64)
                inst.scaler.var_ = np.array(state.get("var", np.square(state["scale"])), dtype=np.float64)
            elif scaler_type == "minmax" and "data_min" in state:
                inst.scaler.data_min_ = np.array(state["data_min"], dtype=np.float64)
                inst.scaler.data_max_ = np.array(state["data_max"], dtype=np.float64)
                inst.scaler.min_ = np.array(state["min"], dtype=np.float64)
                inst.scaler.scale_ = np.array(state["scale"], dtype=np.float64)
        return inst


class SummaryFeatureScaler:
    """Scaler applied strictly to the 18 engineered summary features in Variant B.
    
    Preserves binary visibility flags {0, 1} and normalized RSSI [0, 1], while
    scaling the summary statistics to [0, 1] so that feature magnitudes remain
    balanced across all 1,058 inputs for gradient descent.
    """

    def __init__(self, scaler_type: str = "minmax"):
        self.scaler_type = scaler_type
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0)) if scaler_type == "minmax" else StandardScaler()
        self.summary_columns: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, base_feature_count: int = 1040) -> "SummaryFeatureScaler":
        """Identify and fit on summary columns exceeding base feature count."""
        cols = list(X.columns)
        if len(cols) > base_feature_count:
            self.summary_columns = [c for c in cols if not c.startswith("WAP")]
            if self.summary_columns:
                self.scaler.fit(X[self.summary_columns].values)
                logger.info(
                    "Fitted %s scaler on %d summary features: %s",
                    self.scaler_type,
                    len(self.summary_columns),
                    self.summary_columns[:5],
                )
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform summary features in a copy of DataFrame."""
        if not self.is_fitted or not self.summary_columns:
            return X
        X_out = X.copy()
        matching_cols = [c for c in self.summary_columns if c in X_out.columns]
        if matching_cols:
            X_out[matching_cols] = self.scaler.transform(X_out[matching_cols].values)
        return X_out

    def fit_transform(self, X: pd.DataFrame, base_feature_count: int = 1040) -> pd.DataFrame:
        return self.fit(X, base_feature_count).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scaler to dictionary."""
        if not self.is_fitted:
            return {"is_fitted": False, "scaler_type": self.scaler_type, "summary_columns": []}
        state: Dict[str, Any] = {
            "is_fitted": True,
            "scaler_type": self.scaler_type,
            "summary_columns": self.summary_columns,
        }
        if isinstance(self.scaler, MinMaxScaler) and hasattr(self.scaler, "data_min_"):
            state["data_min"] = self.scaler.data_min_.tolist()
            state["data_max"] = self.scaler.data_max_.tolist()
            state["min"] = self.scaler.min_.tolist()
            state["scale"] = self.scaler.scale_.tolist()
        elif isinstance(self.scaler, StandardScaler) and hasattr(self.scaler, "mean_"):
            state["mean"] = self.scaler.mean_.tolist()
            state["scale"] = self.scaler.scale_.tolist()
        return state

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "SummaryFeatureScaler":
        inst = cls(scaler_type=state.get("scaler_type", "minmax"))
        inst.summary_columns = state.get("summary_columns", [])
        if state.get("is_fitted", False):
            inst.is_fitted = True
            if inst.scaler_type == "minmax" and "data_min" in state:
                inst.scaler.data_min_ = np.array(state["data_min"], dtype=np.float64)
                inst.scaler.data_max_ = np.array(state["data_max"], dtype=np.float64)
                inst.scaler.min_ = np.array(state["min"], dtype=np.float64)
                inst.scaler.scale_ = np.array(state["scale"], dtype=np.float64)
            elif inst.scaler_type == "standard" and "mean" in state:
                inst.scaler.mean_ = np.array(state["mean"], dtype=np.float64)
                inst.scaler.scale_ = np.array(state["scale"], dtype=np.float64)
        return inst


class MultitaskMLP(nn.Module):
    """PyTorch Multi-Layer Perceptron with Shared Backbone and 4 Task Heads.
    
    Shared Trunk:
      - Linear(input_dim -> 512) -> BatchNorm1d(512) -> ReLU -> Dropout(0.30)
      - Linear(512 -> 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.25)
      - Linear(256 -> 128) -> ReLU
      
    Task Heads (branching from 128-dim shared representation):
      - Head 1: Building classification -> Linear(128, 3) (logits)
      - Head 2: Floor classification -> Linear(128, 5) (logits)
      - Head 3: Longitude regression -> Linear(128, 1) (continuous coordinate)
      - Head 4: Latitude regression -> Linear(128, 1) (continuous coordinate)
    """

    def __init__(
        self,
        input_dim: int = 1058,
        hidden_dims: Optional[List[int]] = None,
        dropouts: Optional[List[float]] = None,
        num_buildings: int = 3,
        num_floors: int = 5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        if dropouts is None:
            dropouts = [0.30, 0.25]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropouts = dropouts
        self.num_buildings = num_buildings
        self.num_floors = num_floors

        # Shared Backbone Trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropouts[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropouts[1]),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
        )

        # Output Heads
        self.head_building = nn.Linear(hidden_dims[2], num_buildings)
        self.head_floor = nn.Linear(hidden_dims[2], num_floors)
        self.head_longitude = nn.Linear(hidden_dims[2], 1)
        self.head_latitude = nn.Linear(hidden_dims[2], 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through shared trunk and all four heads.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            
        Returns:
            Dictionary containing:
                'building': Raw logits of shape (batch_size, num_buildings)
                'floor': Raw logits of shape (batch_size, num_floors)
                'longitude': Continuous predictions of shape (batch_size, 1)
                'latitude': Continuous predictions of shape (batch_size, 1)
        """
        shared_repr = self.shared_trunk(x)
        logits_bldg = self.head_building(shared_repr)
        logits_floor = self.head_floor(shared_repr)
        pred_lon = self.head_longitude(shared_repr)
        pred_lat = self.head_latitude(shared_repr)

        return {
            "building": logits_bldg,
            "floor": logits_floor,
            "longitude": pred_lon,
            "latitude": pred_lat,
        }


class MultitaskLoss(nn.Module):
    """Weighted Multitask Loss combining classification and coordinate regression."""

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.loss_weights = loss_weights or {
            "building": 1.0,
            "floor": 2.0,
            "longitude": 1.0,
            "latitude": 1.0,
        }

        self.criterion_bldg = nn.CrossEntropyLoss()
        self.criterion_floor = nn.CrossEntropyLoss()
        if loss_type == "mse":
            self.criterion_coord = nn.MSELoss()
        else:
            self.criterion_coord = nn.SmoothL1Loss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted total loss and individual loss components."""
        l_bldg = self.criterion_bldg(predictions["building"], targets["building"])
        l_floor = self.criterion_floor(predictions["floor"], targets["floor"])
        l_lon = self.criterion_coord(predictions["longitude"], targets["longitude"])
        l_lat = self.criterion_coord(predictions["latitude"], targets["latitude"])

        w_bldg = self.loss_weights.get("building", 1.0)
        w_floor = self.loss_weights.get("floor", 2.0)
        w_lon = self.loss_weights.get("longitude", 1.0)
        w_lat = self.loss_weights.get("latitude", 1.0)

        total_loss = (
            w_bldg * l_bldg
            + w_floor * l_floor
            + w_lon * l_lon
            + w_lat * l_lat
        )

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_building": float(l_bldg.item()),
            "loss_floor": float(l_floor.item()),
            "loss_longitude": float(l_lon.item()),
            "loss_latitude": float(l_lat.item()),
        }

        return total_loss, loss_dict


class IndoorAIMultitaskMLP:
    """High-level Multitask MLP localizer orchestrator for IndoorAI.
    
    Manages:
    - Zero-leakage input validation (rejects target and metadata columns).
    - Summary feature scaling (fitted strictly on training data).
    - Coordinate target standardization (fitted strictly on training data).
    - Model training with early stopping and ReduceLROnPlateau.
    - Inference with automatic coordinate unscaling and building-floor index derivation.
    - Model checkpoint serialization and loading.
    """

    def __init__(
        self,
        params: Optional[MLPHyperparameters] = None,
    ):
        self.params = params or MLPHyperparameters()
        self.feature_names_in_: Optional[List[str]] = None
        self.summary_scaler = (
            SummaryFeatureScaler(scaler_type=self.params.summary_scaler_type)
            if self.params.scale_summary
            else None
        )
        self.coord_scaler = (
            CoordinateStandardizer(scaler_type=self.params.coordinate_scaler_type)
            if self.params.scale_coordinates
            else None
        )
        self.model: Optional[MultitaskMLP] = None
        self.is_fitted_ = False
        self.training_history_: Dict[str, List[float]] = {}

        # Set device
        if self.params.device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif self.params.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _validate_inputs(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Validate input features and enforce zero data leakage."""
        if isinstance(X, pd.DataFrame):
            forbidden = FORBIDDEN_COLUMNS.intersection(set(X.columns))
            if forbidden:
                raise ValueError(
                    f"CRITICAL ZERO-LEAKAGE VIOLATION: Forbidden columns found in feature matrix: {forbidden}"
                )
            return X
        elif isinstance(X, np.ndarray):
            if self.feature_names_in_ is not None:
                if X.shape[1] != len(self.feature_names_in_):
                    raise ValueError(
                        f"Expected {len(self.feature_names_in_)} features, got {X.shape[1]}"
                    )
                return pd.DataFrame(X, columns=self.feature_names_in_)
            return pd.DataFrame(X)
        else:
            raise TypeError(f"Unsupported input feature type: {type(X)}")

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.DataFrame, Dict[str, Any]],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        verbose: bool = True,
    ) -> "IndoorAIMultitaskMLP":
        """Train Multitask MLP with zero-leakage scaling, early stopping, and LR scheduling.
        
        Args:
            X_train: Training feature matrix (1,058 columns).
            y_train: Training target labels/coordinates containing:
                     'BUILDINGID', 'FLOOR', 'LONGITUDE', 'LATITUDE'.
            X_val: Optional validation feature matrix for early stopping.
            y_val: Optional validation targets for early stopping.
            verbose: Whether to log training progress per epoch.
            
        Returns:
            Self (fitted estimator).
        """
        torch.manual_seed(self.params.random_seed)
        np.random.seed(self.params.random_seed)

        # 1. Validate and register feature columns
        df_X = self._validate_inputs(X_train)
        self.feature_names_in_ = list(df_X.columns)
        self.params.input_dim = len(self.feature_names_in_)

        # 2. Scale summary features strictly on training features
        if self.summary_scaler is not None:
            df_X_scaled = self.summary_scaler.fit_transform(df_X)
        else:
            df_X_scaled = df_X

        # 3. Extract and scale training targets
        if isinstance(y_train, pd.DataFrame):
            bldg_train = y_train["BUILDINGID"].values.astype(np.int64)
            floor_train = y_train["FLOOR"].values.astype(np.int64)
            coords_train = y_train[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
        else:
            bldg_train = np.asarray(y_train["building"], dtype=np.int64)
            floor_train = np.asarray(y_train["floor"], dtype=np.int64)
            coords_train = np.column_stack([y_train["longitude"], y_train["latitude"]]).astype(np.float32)

        if self.coord_scaler is not None:
            coords_train_scaled = self.coord_scaler.fit(coords_train).transform(coords_train)
        else:
            coords_train_scaled = coords_train

        lon_train_scaled = coords_train_scaled[:, 0:1].astype(np.float32)
        lat_train_scaled = coords_train_scaled[:, 1:2].astype(np.float32)

        # 4. Prepare Validation data if provided
        has_val = X_val is not None and y_val is not None
        if has_val:
            df_val = self._validate_inputs(X_val)
            if self.summary_scaler is not None:
                df_val_scaled = self.summary_scaler.transform(df_val)
            else:
                df_val_scaled = df_val

            if isinstance(y_val, pd.DataFrame):
                bldg_val = y_val["BUILDINGID"].values.astype(np.int64)
                floor_val = y_val["FLOOR"].values.astype(np.int64)
                coords_val = y_val[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
            else:
                bldg_val = np.asarray(y_val["building"], dtype=np.int64)
                floor_val = np.asarray(y_val["floor"], dtype=np.int64)
                coords_val = np.column_stack([y_val["longitude"], y_val["latitude"]]).astype(np.float32)

            if self.coord_scaler is not None:
                coords_val_scaled = self.coord_scaler.transform(coords_val)
            else:
                coords_val_scaled = coords_val

            lon_val_scaled = coords_val_scaled[:, 0:1].astype(np.float32)
            lat_val_scaled = coords_val_scaled[:, 1:2].astype(np.float32)

            val_tensor_x = torch.tensor(df_val_scaled.values, dtype=torch.float32).to(self.device)
            val_targets = {
                "building": torch.tensor(bldg_val, dtype=torch.long).to(self.device),
                "floor": torch.tensor(floor_val, dtype=torch.long).to(self.device),
                "longitude": torch.tensor(lon_val_scaled, dtype=torch.float32).to(self.device),
                "latitude": torch.tensor(lat_val_scaled, dtype=torch.float32).to(self.device),
            }

        # 5. Build DataLoader
        dataset_train = TensorDataset(
            torch.tensor(df_X_scaled.values, dtype=torch.float32),
            torch.tensor(bldg_train, dtype=torch.long),
            torch.tensor(floor_train, dtype=torch.long),
            torch.tensor(lon_train_scaled, dtype=torch.float32),
            torch.tensor(lat_train_scaled, dtype=torch.float32),
        )
        loader_train = DataLoader(
            dataset_train,
            batch_size=self.params.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # 6. Instantiate Model, Loss, Optimizer, and LR Scheduler
        self.model = MultitaskMLP(
            input_dim=self.params.input_dim,
            hidden_dims=self.params.hidden_dims,
            dropouts=self.params.dropouts,
            num_buildings=self.params.num_buildings,
            num_floors=self.params.num_floors,
        ).to(self.device)

        criterion = MultitaskLoss(
            loss_type=self.params.loss_type,
            loss_weights=self.params.loss_weights,
        ).to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.params.lr_factor,
            patience=self.params.lr_patience,
            min_lr=self.params.min_lr,
        )

        # 7. Training Loop with Early Stopping
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_loss_bldg": [],
            "train_loss_floor": [],
            "train_loss_lon": [],
            "train_loss_lat": [],
            "val_loss": [],
            "val_loss_bldg": [],
            "val_loss_floor": [],
            "val_loss_lon": [],
            "val_loss_lat": [],
            "val_acc_bldg": [],
            "val_acc_floor": [],
            "lr": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None

        logger.info(
            "Starting Multitask MLP training: %d samples, input_dim=%d, batch_size=%d, max_epochs=%d, device=%s",
            len(df_X),
            self.params.input_dim,
            self.params.batch_size,
            self.params.epochs,
            self.device,
        )

        for epoch in range(1, self.params.epochs + 1):
            self.model.train()
            running_loss = 0.0
            running_bldg = 0.0
            running_floor = 0.0
            running_lon = 0.0
            running_lat = 0.0
            n_batches = 0

            for bx, bb, bf, blon, blat in loader_train:
                bx = bx.to(self.device)
                targets_batch = {
                    "building": bb.to(self.device),
                    "floor": bf.to(self.device),
                    "longitude": blon.to(self.device),
                    "latitude": blat.to(self.device),
                }

                optimizer.zero_grad()
                preds = self.model(bx)
                loss, loss_comps = criterion(preds, targets_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss_comps["loss_total"]
                running_bldg += loss_comps["loss_building"]
                running_floor += loss_comps["loss_floor"]
                running_lon += loss_comps["loss_longitude"]
                running_lat += loss_comps["loss_latitude"]
                n_batches += 1

            train_loss = running_loss / max(n_batches, 1)
            history["train_loss"].append(train_loss)
            history["train_loss_bldg"].append(running_bldg / max(n_batches, 1))
            history["train_loss_floor"].append(running_floor / max(n_batches, 1))
            history["train_loss_lon"].append(running_lon / max(n_batches, 1))
            history["train_loss_lat"].append(running_lat / max(n_batches, 1))
            current_lr = float(optimizer.param_groups[0]["lr"])
            history["lr"].append(current_lr)

            # Validation evaluation
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(val_tensor_x)
                    val_loss, val_comps = criterion(val_preds, val_targets)
                    v_loss_val = val_comps["loss_total"]

                    # Accuracy
                    pred_bldg_cls = val_preds["building"].argmax(dim=1)
                    pred_floor_cls = val_preds["floor"].argmax(dim=1)
                    acc_bldg = float((pred_bldg_cls == val_targets["building"]).float().mean().item())
                    acc_floor = float((pred_floor_cls == val_targets["floor"]).float().mean().item())

                history["val_loss"].append(v_loss_val)
                history["val_loss_bldg"].append(val_comps["loss_building"])
                history["val_loss_floor"].append(val_comps["loss_floor"])
                history["val_loss_lon"].append(val_comps["loss_longitude"])
                history["val_loss_lat"].append(val_comps["loss_latitude"])
                history["val_acc_bldg"].append(acc_bldg)
                history["val_acc_floor"].append(acc_floor)

                scheduler.step(v_loss_val)

                if v_loss_val < best_val_loss:
                    best_val_loss = v_loss_val
                    patience_counter = 0
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if verbose and (epoch % 5 == 0 or epoch == 1 or patience_counter >= self.params.patience):
                    logger.info(
                        "Epoch %3d/%3d | Train Loss: %.4f | Val Loss: %.4f | Bldg Acc: %.2f%% | Floor Acc: %.2f%% | LR: %.1e",
                        epoch,
                        self.params.epochs,
                        train_loss,
                        v_loss_val,
                        acc_bldg * 100.0,
                        acc_floor * 100.0,
                        current_lr,
                    )

                if patience_counter >= self.params.patience:
                    logger.info("Early stopping triggered at epoch %d (patience=%d)", epoch, self.params.patience)
                    break
            else:
                if verbose and (epoch % 10 == 0 or epoch == 1):
                    logger.info("Epoch %3d/%3d | Train Loss: %.4f | LR: %.1e", epoch, self.params.epochs, train_loss, current_lr)

        # Restore best weights if validation was performed
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info("Restored best model weights with validation loss: %.4f", best_val_loss)

        self.training_history_ = history
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """Perform full multi-target inference.
        
        Args:
            X: Input feature matrix (1,058 columns).
            
        Returns:
            Dictionary containing:
                'building': Predicted building class array (0, 1, 2)
                'floor': Predicted floor class array (0, 1, 2, 3, 4)
                'building_floor': Mapped building-floor index array (0 to 12)
                'building_floor_str': Mapped string representation (e.g. '0_0')
                'longitude': Unscaled longitude coordinates in original units
                'latitude': Unscaled latitude coordinates in original units
        """
        if not self.is_fitted_ or self.model is None:
            raise ValueError("IndoorAIMultitaskMLP is not fitted.")

        df_X = self._validate_inputs(X)
        if self.summary_scaler is not None:
            df_X_scaled = self.summary_scaler.transform(df_X)
        else:
            df_X_scaled = df_X

        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(df_X_scaled.values, dtype=torch.float32).to(self.device)
            preds = self.model(tensor_x)

            # Classifications
            bldg_preds = preds["building"].argmax(dim=1).cpu().numpy().astype(np.int64)
            floor_preds = preds["floor"].argmax(dim=1).cpu().numpy().astype(np.int64)

            # Coordinates
            lon_scaled = preds["longitude"].cpu().numpy()
            lat_scaled = preds["latitude"].cpu().numpy()

        coords_scaled = np.column_stack([lon_scaled, lat_scaled])
        if self.coord_scaler is not None:
            coords_orig = self.coord_scaler.inverse_transform(coords_scaled)
        else:
            coords_orig = coords_scaled

        pred_lon = coords_orig[:, 0]
        pred_lat = coords_orig[:, 1]

        # Derive Building-Floor Joint Target
        bf_strs = [f"{b}_{f}" for b, f in zip(bldg_preds, floor_preds)]
        bf_indices = np.array([BUILDING_FLOOR_MAP.get(s, -1) for s in bf_strs], dtype=np.int64)

        return {
            "building": bldg_preds,
            "floor": floor_preds,
            "building_floor": bf_indices,
            "building_floor_str": np.array(bf_strs),
            "longitude": pred_lon,
            "latitude": pred_lat,
        }

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """Predict class probabilities for classification heads."""
        if not self.is_fitted_ or self.model is None:
            raise ValueError("IndoorAIMultitaskMLP is not fitted.")

        df_X = self._validate_inputs(X)
        if self.summary_scaler is not None:
            df_X_scaled = self.summary_scaler.transform(df_X)
        else:
            df_X_scaled = df_X

        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(df_X_scaled.values, dtype=torch.float32).to(self.device)
            preds = self.model(tensor_x)

            prob_bldg = torch.softmax(preds["building"], dim=1).cpu().numpy()
            prob_floor = torch.softmax(preds["floor"], dim=1).cpu().numpy()

        return {
            "building": prob_bldg,
            "floor": prob_floor,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save complete model checkpoint, scalers, and hyperparameters to file."""
        if not self.is_fitted_ or self.model is None:
            raise ValueError("Cannot save an unfitted model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "state_dict": self.model.state_dict(),
            "params": self.params.to_dict(),
            "feature_names_in": self.feature_names_in_,
            "summary_scaler": self.summary_scaler.to_dict() if self.summary_scaler else None,
            "coord_scaler": self.coord_scaler.to_dict() if self.coord_scaler else None,
            "training_history": self.training_history_,
        }
        torch.save(payload, path)
        logger.info("Saved Multitask MLP model checkpoint to %s", path)

    @classmethod
    def load(cls, filepath: Union[str, Path], device: Optional[str] = None) -> "IndoorAIMultitaskMLP":
        """Load model checkpoint from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = torch.load(path, map_location="cpu")
        params_dict = payload["params"]
        if device is not None:
            params_dict["device"] = device

        params = MLPHyperparameters(**params_dict)
        inst = cls(params=params)
        inst.feature_names_in_ = payload["feature_names_in"]
        inst.training_history_ = payload.get("training_history", {})

        if payload.get("summary_scaler"):
            inst.summary_scaler = SummaryFeatureScaler.from_dict(payload["summary_scaler"])
        if payload.get("coord_scaler"):
            inst.coord_scaler = CoordinateStandardizer.from_dict(payload["coord_scaler"])

        inst.model = MultitaskMLP(
            input_dim=params.input_dim,
            hidden_dims=params.hidden_dims,
            dropouts=params.dropouts,
            num_buildings=params.num_buildings,
            num_floors=params.num_floors,
        ).to(inst.device)

        inst.model.load_state_dict(payload["state_dict"])
        inst.is_fitted_ = True
        logger.info("Loaded Multitask MLP from %s onto %s", path, inst.device)
        return inst

