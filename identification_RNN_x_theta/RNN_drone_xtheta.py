# Standard libraries
import math                                                 # Mathematical functions (sin, cos, sqrt, etc.)
import random                                               # Random number generation and sampling
import sys                                                  # System-specific parameters and functions (e.g. exit, argv)
import os                                                   # Operating system interfaces (file paths, environment variables)
import time                                                 # Elapsed time of the training
import json                                                 # Trace of each run (run_info.json)
import warnings                                             # Filter irrelevant library warnings

# Scientific computing and data handling
import numpy as np                                          # Numerical computing (arrays, linear algebra, etc.)
import matplotlib.pyplot as plt                             # Plotting and visualization
import csv                                                  # Reading and writing CSV files
import pandas as pd                                         # Data manipulation and analysis (DataFrames, Series)

# Python type hints and data structures
from dataclasses import dataclass                           # Decorator for creating classes with auto-generated special methods
from typing import Tuple                                    # Type hints for tuples and other complex types

# PyTorch: Deep learning framework
import torch                                                # Tensors and neural network building blocks
from torch import nn                                        # Neural network layers and loss functions
from torch.utils.data import Dataset, DataLoader            # Dataset and batch loading utilities

# PyTorch Lightning: High-level interface for PyTorch
import lightning as L                                       # Core Lightning module
from lightning.pytorch import Trainer                       # Training loop management
from lightning.pytorch.callbacks import (                   # Training utilities:
    EarlyStopping,                                          # Stop training when a metric stops improving
    ModelCheckpoint,                                        # Save the best model during training
    TQDMProgressBar                                         # Progress bar for training
)
from lightning.pytorch.loggers import TensorBoardLogger     # Logging to TensorBoard

# Metrics for model evaluation
import torchmetrics as tm                                   # Collection of metrics (RMSE, Accuracy, etc.) for PyTorch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# Work in the folder of this file (data files and checkpoints are next to it)
if '__file__' in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Hide Lightning warnings that are not relevant here (small dataset trained on CPU)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*LeafSpec.*")

# Output modelled by this run (x / theta dynamics) and size of the network:
#   python RNN_drone_xtheta.py atheta 16   or   python RNN_drone_xtheta.py ax 32
TARGET = sys.argv[1] if len(sys.argv) > 1 else 'atheta'
HIDDEN = int(sys.argv[2]) if len(sys.argv) > 2 else 16
UNIT = 'rad/s²' if TARGET == 'atheta' else 'm/s²'



def load_csv_to_tensors(input_csv: str, target_csv: str, input_size: int = 1, output_size: int = 1, has_header: bool = False):
    """
    Load two CSV files (inputs U and outputs Y) and convert them into PyTorch tensors.
    
    Assumptions:
      - Each row in the CSV represents a full sequence (trajectory)
      - Columns represent time steps in a 'wide' format
      - input_size and output_size define the number of features per time step
    
    Args:
        input_csv (str): Path to the CSV file containing input sequences (U)
        target_csv (str): Path to the CSV file containing output sequences (Y)
        input_size (int): Number of input features per time step
        output_size (int): Number of output features per time step
        has_header (bool): True if the CSV files include a header row
    
    Returns:
        X (torch.Tensor): Input tensor of shape (N, T, input_size)
        Y (torch.Tensor): Output tensor of shape (N, T, output_size)
    """
    
    # Determine header option for pandas
    header = 0 if has_header else None
    
    # Load CSV files into NumPy arrays (float32 for PyTorch compatibility)
    U = pd.read_csv(input_csv, header=header).values.astype('float32')
    Y = pd.read_csv(target_csv, header=header).values.astype('float32')
    
    # Check that both files have the same number of sequences
    if U.shape[0] != Y.shape[0]:
        raise ValueError(f"Number of sequences differs: U has {U.shape[0]}, Y has {Y.shape[0]}.")
    
    # Helper function to reshape from (N, T*dim) to (N, T, dim)
    def reshape(arr, dim):
        if arr.shape[1] % dim != 0:
            raise ValueError(f"Number of columns {arr.shape[1]} is not divisible by dim={dim}.")
        T = arr.shape[1] // dim
        return arr.reshape(arr.shape[0], T, dim)
    
    # Reshape both input and output arrays
    U = reshape(U, input_size)
    Y = reshape(Y, output_size)
    
    # Convert to PyTorch tensors
    X = torch.tensor(U, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    
    return X, Y


class dataModule(L.LightningDataModule):
    """
    LightningDataModule for handling dataset splitting and loading.
    This class prepares the data for training, validation, and testing in PyTorch Lightning.
    """
    def __init__(self, data_u, data_y, batch_size=32):
        """
        Initialize the data module with input and output data.

        Args:
            data_u (torch.Tensor): Input data tensor.
            data_y (torch.Tensor): Output data tensor.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
        """
        super().__init__()
        self.batch_size = batch_size
        self.N = data_u.shape[0]  # Total number of samples
        self.data_u = data_u  # Store input data
        self.data_y = data_y  # Store output data

    def setup(self, stage=None):
        """
        Split the dataset into training, validation, and test sets.

        Args:
            stage (str, optional): Stage of the training process (fit, validate, test, predict). Defaults to None.
        """

        self.N_train = int(0.70 * self.N)
        self.N_val = int(0.85 * self.N) 
        self.N_test = self.N

        self.train_dataset = torch.utils.data.TensorDataset(self.data_u[:self.N_train], self.data_y[:self.N_train])
        self.val_dataset = torch.utils.data.TensorDataset(self.data_u[self.N_train:self.N_val], self.data_y[self.N_train:self.N_val])
        self.test_dataset = torch.utils.data.TensorDataset(self.data_u[self.N_val:self.N_test], self.data_y[self.N_val:self.N_test])
                            #% To be completed

        # Create TensorDatasets for each split
        # Train set
        #self.train_dataset = torch.utils.data.TensorDataset(self.N_train)

        # Vaidation set
        #self.val_dataset = torch.utils.data.TensorDataset(self.N_val)

        
        # Test set
        #self.test_dataset = torch.utils.data.TensorDataset(self.N_test)

    def train_dataloader(self):
        """
        Return the DataLoader for the training set.

        Returns:
            DataLoader: Training data loader with shuffling.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Return the DataLoader for the validation set.

        Returns:
            DataLoader: Validation data loader without shuffling.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """
        Return the DataLoader for the test set.

        Returns:
            DataLoader: Test data loader without shuffling.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)



class model_nn(L.LightningModule):
    """
    Sequential neural network model (RNN/LSTM/GRU) for aligned sequence-to-sequence mapping U->Y.
    """
    def __init__(
        self,
        input_size: int = 1 ,        # To be completed
        output_size: int = 1 ,       # To be completed
        rnn_type: str = "LSTM",  # "RNN", "LSTM", "GRU"
        hidden_size: int = 16 ,       # To be completed
        num_layers: int = 1 ,        # To be completed
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Initialize the model with the specified architecture and hyperparameters.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            rnn_type (str): Type of recurrent layer ("RNN", "LSTM", "GRU").
            hidden_size (int): Number of hidden units in the RNN layer.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout probability between RNN layers.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay (L2 penalty) for the optimizer.
            bidirectional (bool): If True, becomes a bidirectional RNN.
        """
        super().__init__()
        self.save_hyperparameters()
        # Select and initialize the RNN type
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type.upper() == "RNN":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity='tanh',
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError("rnn_type must be one of {'RNN','LSTM','GRU'}")
        # Linear head for output projection
        d = 2 if bidirectional else 1
        self.head = nn.Linear(d * hidden_size, output_size)
        self.loss_fn = nn.MSELoss()  # Mean squared error loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, T, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch, T, output_size).
        """
        h, _ = self.rnn(x)              # h: (batch, T, d*hidden_size)
        y = self.head(h)                # y: (batch, T, output_size)
        return y

    def training_step(self, batch, batch_idx):
        """
        Training step: forward pass, loss computation, and logging.

        Args:
            batch: Tuple of (input, target) tensors.
            batch_idx: Index of the current batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step: forward pass, loss computation, and logging.

        Args:
            batch: Tuple of (input, target) tensors.
            batch_idx: Index of the current batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Test step: forward pass, loss computation, and logging.

        Args:
            batch: Tuple of (input, target) tensors.
            batch_idx: Index of the current batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer instance.
        """
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return opt


class EpochLogger(L.Callback):
    """
    Print one line per epoch in the terminal (kept in the history, unlike the progress bar):
    train / validation loss, validation RMSE in m/s², best epoch so far and elapsed time.
    """
    def __init__(self, y_scale=1.0):
        super().__init__()
        self.y_scale = y_scale                  # Normalised output -> m/s²

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.time()
        self.best = float('inf')
        self.best_epoch = 0
        pl_module.print(f"{'epoch':>9} | {'train_loss':>10} | {'val_loss':>10} | {'val RMSE':>12} | {'best val RMSE':>23} | {'time':>6}")

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'val_loss' not in metrics or 'train_loss_epoch' not in metrics:
            return
        epoch = trainer.current_epoch + 1
        train_loss = metrics['train_loss_epoch'].item()
        val_loss = metrics['val_loss'].item()
        improved = val_loss < self.best
        if improved:
            self.best, self.best_epoch = val_loss, epoch
        rmse = math.sqrt(val_loss) * self.y_scale         # RMSE in m/s² (MSE is computed on normalised data)
        best_rmse = math.sqrt(self.best) * self.y_scale
        pl_module.print(f"{epoch:4d}/{trainer.max_epochs:<4d} | {train_loss:10.3e} | {val_loss:10.3e} | {rmse:7.3f} {UNIT} | "
                        f"{best_rmse:7.3f} {UNIT} (ep {self.best_epoch:4d}) | {time.time() - self.t0:5.0f} s" + ("  *" if improved else ""))


def train_run(data_u, data_y, model_save_name, epoch_max=10, nb_train=1, patience_EarlyStopping=200, refresh_rate_brogress_bar=10, batch_size = 32, model_kwargs=None, y_scale=1.0, seed=0):
    """
    Train a model multiple times with the given data and save the results.

    Args:
        data_u (np.ndarray): Input data for training.
        data_y (np.ndarray): Output data for training.
        model_save_name (str): Name of the directory to save model logs and checkpoints.
        epoch_max (int, optional): Maximum number of training epochs. Defaults to 10.
        nb_train (int, optional): Number of training runs. Defaults to 1.
        patience_EarlyStopping (int, optional): Patience for early stopping. Defaults to 200.
        refresh_rate_brogress_bar (int, optional): Refresh rate for the progress bar. Defaults to 10.
        model_kwargs (dict, optional): Hyperparameters passed to model_nn (rnn_type, hidden_size, ...).
        y_scale (float, optional): Output normalisation scale, to print the validation RMSE in m/s². Defaults to 1.
        seed (int, optional): Random seed of the first run (run i uses seed + i). Defaults to 0.

    Returns:
        list: One dict per training run (best checkpoint, log directory, best val_loss and epoch, training time...).
    """
    runs = []
    for index_train in range(nb_train):
        print("Train ", index_train)
        L.seed_everything(seed + index_train)       # Different (and reproducible) initial weights and batch order for each run

        # Initialize the data module with input and output data
        data_module = dataModule(data_u, data_y, batch_size=batch_size)

        # Initialize the model
        model = model_nn(**(model_kwargs or {}))

        # Set up TensorBoard logger to save training logs
        logger = L.pytorch.loggers.TensorBoardLogger(model_save_name)

        # Early stopping callback to halt training if validation loss does not improve
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=patience_EarlyStopping,
            mode='min'
        )

        # Model checkpoint callback to save the best model based on validation loss
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',  # Metric to monitor for saving the best model
            dirpath=logger.log_dir,  # Directory to save the model checkpoints
            filename='model-{epoch:02d}-{val_loss:.2e}',  # Filename format for saved models
            save_top_k=1,  # Save only the best model
            mode='min',  # Save the model with the minimum validation loss
        )

        # Initialize the trainer with callbacks and logger
        trainer = L.Trainer(
            max_epochs=epoch_max,
            logger=logger,
            log_every_n_steps=1,                # Log every batch (few batches per epoch)
            callbacks=[
                early_stopping_callback,
                checkpoint_callback,
                TQDMProgressBar(refresh_rate=refresh_rate_brogress_bar),
                EpochLogger(y_scale=y_scale)    # One line per epoch in the terminal
            ]
        )

        # Train the model
        t_start = time.time()
        trainer.fit(model, data_module)
        train_time = time.time() - t_start

        # Test the model after training
        test_results = trainer.test(model, datamodule=data_module)

        # Print the logger information
        print(logger)
        best_path = checkpoint_callback.best_model_path
        runs.append(dict(
            seed=seed + index_train,                                       # Random seed of this run
            checkpoint=best_path.replace(os.sep, '/'),                    # Best model of this run
            log_dir=logger.log_dir,                                        # model_rnn/lightning_logs/version_N
            best_val_loss=float(checkpoint_callback.best_model_score),
            best_epoch=int(torch.load(best_path, map_location='cpu', weights_only=False)['epoch']) + 1,
            epochs_run=int(trainer.current_epoch),
            train_time_s=round(train_time, 1),
            test_loss_last_epoch=float(test_results[0]['test_loss']),      # trainer.test uses the last epoch model
        ))
    return runs


# Cleaned datasets (data_base_cleaning/clean_data.py): 1 row = 1 trial of 2000 steps, gzip read by pandas
csv_u = '../generation_donnees_x_theta/bdd_in_utheta.csv.gz'          # Input data (pitch command u_theta)
csv_y = f'../generation_donnees_x_theta/bdd_out_{TARGET}.csv.gz'     # Output data (a_theta [rad/s²] or ax [m/s²])
Te = 0.05                                   # Sampling period [s]

# Define the dimensions of input and output for the model
input_size =   1                            # Number of input features per time step         # To be completed
output_size =   1                           # Number of output features per time step        # To be completed
has_header = False                          # Flag indicating whether the CSV files have a header row

# Load the input and output data from CSV files into PyTorch tensors
tensor_u, tensor_y = load_csv_to_tensors(
    csv_u,                                  # Path to the input CSV file
    csv_y,                                  # Path to the output CSV file
    input_size,                             # Number of input features
    output_size,                            # Number of output features
    has_header                              # Whether the CSV files have a header row
)

# Normalisation: the network works on quantities of order 1 (u_theta in [-0.021, 0.021], outputs equal to 0 at rest)
# The output is centred on its rest value (az = -g for u = 0), so that az_n is in [-1, 1]
u_scale = 0.02                              # Input scale [-] (range of u_theta in the dataset)
y_offset = 0.0                              # Output offset (rest value: 0)
y_scale = 2.6 if TARGET == 'atheta' else 4.5   # Output scale (range of a_theta [rad/s²] / ax [m/s²])
tensor_u_n = tensor_u / u_scale
tensor_y_n = (tensor_y - y_offset) / y_scale


# Define the name for saving the model and logs
model_save_name = f"RNN_{'a_theta' if TARGET == 'atheta' else 'ax'}_{HIDDEN}neurones"             # Directory name where model checkpoints and logs will be saved

# Hyperparameters of the network (see model_nn) - first naive baseline: simple RNN
model_kwargs = dict(
    rnn_type="RNN",                         # "RNN", "LSTM", "GRU"
    hidden_size=HIDDEN,                     # Number of hidden units (command-line argument, 16 by default)
    num_layers=1,                           # Number of stacked recurrent layers
    dropout=0.0,                            # Dropout between layers (only if num_layers > 1)
    lr=3e-3,                                # Learning rate
    weight_decay=0.0,                       # L2 penalty
)

# Training parameters
train_kwargs = dict(
    epoch_max=30,                           # Maximum number of training epochs
    nb_train=3,                             # Number of training runs (same settings, different initial weights)
    seed=1,                                 # Seed of the first run
    patience_EarlyStopping=10,              # Number of epochs to wait before early stopping if no improvement
    refresh_rate_brogress_bar=10,           # Refresh rate for the progress bar (in batches)
    batch_size=32,                          # Number of sequences per batch
)

TRAIN = True                                # False: skip training and display the checkpoint given in model_path

if TRAIN:
    # Run the training process with the specified parameters
    runs = train_run(
        tensor_u_n,                             # Input data tensor (normalised)
        tensor_y_n,                             # Output data tensor (normalised)
        model_save_name,                        # Name for saving the model
        model_kwargs=model_kwargs,              # Hyperparameters of the network
        y_scale=y_scale,                        # To print the validation RMSE in m/s²
        **train_kwargs                          # Training parameters
    )

    # Trace of each run in its folder: data, normalisation, hyperparameters, results
    for run in runs:
        run_info = dict(
            version=os.path.basename(run['log_dir']),
            date=time.strftime('%Y-%m-%d %H:%M'),
            data=dict(csv_u=csv_u, csv_y=csv_y, source='generation_donnees_x_theta/generer_bdd_x_theta.m (UAV_model.slx, pitch loop)', Te=Te,
                      n_sequences=int(tensor_u.shape[0]), n_samples=int(tensor_u.shape[1]), split=[0.70, 0.15, 0.15]),
            normalization=dict(u_scale=u_scale, y_offset=y_offset, y_scale=y_scale),
            model=dict(model_kwargs, input_size=input_size, output_size=output_size),
            training=dict(train_kwargs, **{k: run[k] for k in ('seed', 'best_epoch', 'epochs_run', 'best_val_loss',
                                                                 'train_time_s', 'test_loss_last_epoch')}),
            checkpoint=run['checkpoint'],
        )
        with open(os.path.join(run['log_dir'], 'run_info.json'), 'w') as f:
            json.dump(run_info, f, indent=2)

    # Comparison of all the models trained so far (run_info.json of each version). The best one is chosen
    # on the validation loss: the test set stays unseen, for the final evaluation below
    logs_dir = os.path.join(model_save_name, 'lightning_logs')
    print(f"\n{'version':>10} | {'seed':>4} | {'best epoch':>10} | {'val RMSE':>12}")
    best_val_loss = float('inf')
    for version in sorted(os.listdir(logs_dir), key=lambda v: int(v.split('_')[-1])):
        info_path = os.path.join(logs_dir, version, 'run_info.json')
        if not os.path.isfile(info_path):
            continue
        with open(info_path) as f:
            training = json.load(f)['training']
        print(f"{version:>10} | {str(training.get('seed', '-')):>4} | {training['best_epoch']:4d} / {training['epochs_run']:<3d} | "
              f"{math.sqrt(training['best_val_loss']) * y_scale:7.3f} {UNIT}")
        if training['best_val_loss'] < best_val_loss:
            best_val_loss, best_version = training['best_val_loss'], version
    ckpt = [f for f in os.listdir(os.path.join(logs_dir, best_version)) if f.endswith('.ckpt')][0]
    model_path = os.path.join(logs_dir, best_version, ckpt).replace(os.sep, '/')    # Best checkpoint of all the runs
    print(f"Best model (validation): {best_version}\n")
else:
    model_path = {('atheta', 16): 'RNN_a_theta_16neurones/lightning_logs/version_2/model-epoch=29-val_loss=6.22e-06.ckpt',
                  ('ax', 16): 'RNN_ax_16neurones/lightning_logs/version_0/model-epoch=28-val_loss=3.39e-02.ckpt',
                  ('ax', 32): 'RNN_ax_32neurones/lightning_logs/version_1/model-epoch=29-val_loss=2.25e-02.ckpt'}[(TARGET, HIDDEN)]   # Best checkpoint of 3 runs


# Load the trained model
model = model_nn.load_from_checkpoint(model_path, map_location=device)                      # Load the model from checkpoint
model.to(device)                                                                            # Move the model to the appropriate device (CPU/GPU)
model.eval()                                                                                # Set the model to evaluation mode
print(model_path)
print(model)                                                                                # Print model architecture and parameters

# Display detailed information about model parameters
for name, param in model.named_parameters():
    print(name, param.shape)                                                                # Print parameter names and shapes


def predict(model, u):
    """
    RNN prediction in physical units (normalisation handled here).

    Args:
        model (model_nn): Trained model.
        u (np.ndarray): Motor command [-], shape (T,) or (N, T).

    Returns:
        np.ndarray: Predicted vertical acceleration az [m/s²], same shape as u.
    """
    u_in = torch.as_tensor(np.atleast_2d(u), dtype=torch.float32).unsqueeze(-1).to(device)  # (N, T, 1)
    with torch.no_grad():
        y_n = model(u_in / u_scale).squeeze(-1).cpu().numpy()                               # Normalised prediction
    y = y_n * y_scale + y_offset
    return y[0] if np.ndim(u) == 1 else y


def metrics(y_true, y_hat):
    """RMSE [m/s²], NRMSE (RMSE / range of y_true) and R^2 over all samples."""
    err = y_hat - y_true
    rmse = np.sqrt(np.mean(err**2))
    nrmse = rmse / (y_true.max() - y_true.min())
    r2 = 1 - np.sum(err**2) / np.sum((y_true - y_true.mean())**2)
    return rmse, nrmse, r2


# Test set: same split as dataModule (last 15 % of the sequences, never seen during training)
i_test = int(0.85 * tensor_u.shape[0])
U_test = tensor_u[i_test:].squeeze(-1).numpy()                                              # Inputs [-]
Y_test = tensor_y[i_test:].squeeze(-1).numpy()                                              # Measured az (noisy) [m/s²]
Y_pred = predict(model, U_test)                                                             # RNN prediction [m/s²]

print(f"\nTest set: {U_test.shape[0]} sequences")
rmse, nrmse, r2 = metrics(Y_test, Y_pred)
print(f"RNN vs measured   RMSE = {rmse:7.3f} {UNIT}   NRMSE = {nrmse:6.4f}   R2 = {r2:.5f}")

# Plot one test sequence (here, the last one): input, measured output and RNN
t = np.arange(U_test.shape[1]) * Te                                                         # Time array [s]
fig, (ax_u, ax_y) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[1, 2])
ax_u.step(t, U_test[-1], where='post', color='#52514e')                                     # Input held between samples
ax_u.set_ylabel("u [-]")
ax_u.set_title("Test sequence", loc='left')
ax_u.grid(True, alpha=0.3)
ax_y.plot(t, Y_test[-1], '.', label="Simulator (dataset)", color='#52514e', markersize=3)
ax_y.plot(t, Y_pred[-1], label="RNN prediction", color='#2a78d6', linewidth=2, linestyle='--')
ax_y.set_xlabel("Time [s]")
ax_y.set_ylabel(f"{TARGET} [{UNIT}]")
ax_y.legend()
ax_y.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'figures/{TARGET}_{HIDDEN}neurones_test_sequence.png', dpi=130)

# Generalisation: inputs never seen during training (steps from rest, as in the dataset: u = 0 for 40 steps)
nb_samples = U_test.shape[1]
step_amps = np.array([-1.0, -0.5, -0.2, 0.2, 0.5, 1.0]) * u_scale                                  # Step magnitudes [-]
U_steps = np.zeros((step_amps.size, nb_samples))
U_steps[:, 40:] = step_amps[:, None]                                                        # Steps applied at t = 40 Te
Y_steps_pred = predict(model, U_steps)

u_static = np.linspace(-1, 1, 21) * u_scale                                                         # Static characteristic az_eq(u)
U_static = np.zeros((u_static.size, nb_samples))
U_static[:, 40:] = u_static[:, None]
az_static_pred = predict(model, U_static)[:, -1]                                            # RNN value at the end of a long step

fig, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(12, 5))
for i, amp in enumerate(step_amps):
    ax_s.plot(t, Y_steps_pred[i], color='#2a78d6', linewidth=2, label="RNN" if i == 0 else None)
    ax_s.annotate(f"u = {amp:+.3f}", (t[-1], Y_steps_pred[i, -1]), xytext=(4, 0),
                  textcoords='offset points', va='center', fontsize=9, color='#52514e')
ax_s.set_title("Step responses", loc='left')
ax_s.set_xlabel("Time [s]")
ax_s.set_ylabel(f"{TARGET} [{UNIT}]")
ax_s.legend()
ax_s.grid(True, alpha=0.3)
ax_c.plot(u_static, az_static_pred, 'o', color='#2a78d6', markersize=6, label=f"RNN (az at t = {t[-1]:.0f} s)")
ax_c.set_title("Static characteristic", loc='left')
ax_c.set_xlabel("u [-]")
ax_c.set_ylabel(f"{TARGET}_eq [{UNIT}]")
ax_c.legend()
ax_c.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'figures/{TARGET}_{HIDDEN}neurones_steps_static.png', dpi=130)
plt.show()


# Extracting weights
weight = model.state_dict()
weight_ih_l0 = weight['rnn.weight_ih_l0'].cpu().numpy()
weight_hh_l0 = weight['rnn.weight_hh_l0'].cpu().numpy()
bias_h = weight['rnn.bias_ih_l0'].cpu().numpy() + weight['rnn.bias_hh_l0'].cpu().numpy()
head_bias = weight['head.bias'].cpu().numpy()
head_weight = weight['head.weight'].cpu().numpy()

# Create the export directory if it does not exist (one per run, next to its checkpoint)
export_dir = os.path.join(os.path.dirname(model_path), 'export_NN')
os.makedirs(export_dir, exist_ok=True)

# Function to write a numpy array in MATLAB format
def write_matlab_array(file_path, array):
    with open(file_path, 'w') as f:
        if array.ndim == 0:  # Scalar
            f.write(f"[{array}]")
        elif array.ndim == 1:  # 1D array
            f.write('[' + ' '.join(map(str, array)) + ']')
        else:  # 2D array
            mat_str = '[' + '; '.join([' '.join(map(str, row)) for row in array]) + ']'
            f.write(mat_str)

# Write each weight to a separate file
write_matlab_array(os.path.join(export_dir, 'weight_ih_l0.txt'), weight_ih_l0)
write_matlab_array(os.path.join(export_dir, 'weight_hh_l0.txt'), weight_hh_l0)
write_matlab_array(os.path.join(export_dir, 'bias_h.txt'), bias_h)
write_matlab_array(os.path.join(export_dir, 'head_bias.txt'), head_bias)
write_matlab_array(os.path.join(export_dir, 'head_weight.txt'), head_weight)
write_matlab_array(os.path.join(export_dir, 'normalization.txt'), np.array([u_scale, y_offset, y_scale]))  # u_n = u/u_scale, az = y_n*y_scale + y_offset

print(f"Export completed! Files are in MATLAB format in {export_dir}")