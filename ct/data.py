from pathlib import Path
from typing import NamedTuple

import numpy as np
from astropy import units as u

CT_SHAPE = (2976, 2464)
OFFSET = 2048  # Offset in bytes for raw CT data

class CTStats(NamedTuple):
    mean: float
    variance: float
    std: float

class A62(NamedTuple):
    U: float  # Anodenspannung
    I: float  # Anodenstrom
    tau: float  # Integrationszeit
    date: str  # Aufnahmedatum
    series_name: str  # Serienname
    data: np.ndarray  # CT Scan
    stats: CTStats  # Computed statistics

class A63(NamedTuple):
    U: float  # Anodenspannung
    I: float  # Anodenstrom
    tau: float  # Integrationszeit
    series_name: str  # Serienname
    filter: bool # Whether a filter was used
    data: np.ndarray  # CT Scan

def ct_to_stats(data: np.ndarray) -> tuple[float, float, float]:
    """
    Compute mean, variance and standard deviation of CT data.
    """
    mean_val = np.mean(data)
    variance_val = np.var(data)
    std = np.std(data)
    return mean_val, variance_val, std

def load_A621(filepath: Path) -> A62:
    filename = filepath.name
    series_name, U_str, I_str, tau_str, date_str = filename.split("_")
    U = float(U_str.replace("kV", "")) * u.kV
    I = float(I_str.replace("muA", "").replace("uA", "")) * u.uA
    tau = float(tau_str.replace("s", "")) * u.s
    date = date_str

    # Load data and make relative to 0-16bit
    data = np.fromfile(filepath, dtype=np.uint16, offset=OFFSET).reshape(CT_SHAPE)
    data = data.astype(np.float64) / (2**16 - 1)  #

    stats = CTStats(*ct_to_stats(data))

    return A62(U, I, tau, date, series_name, data, stats)


def load_A62(folderpaths: list[Path]) -> list[A62]:
    """
    Load all A6.2 datasets from given folder paths.
    """
    datasets = []
    for folderpath in folderpaths:
        for filepath in folderpath.glob("*.raw"):
            dataset = load_A621(filepath)
            datasets.append(dataset)
    return datasets


def load_A63x(filepath: Path) -> A63:
    filename = filepath.name
    series_name, U_str, I_str, tau_str = filename.split("_")[:4]
    U = float(U_str.replace("kV", "")) * u.kV
    I = float(I_str.replace("muA", "").replace("uA", "")) * u.uA
    tau = float(tau_str.replace("s", "")) * u.s
    filter_ = "kein_Filter" not in filename

    # Load data and make relative to 0-16bit
    data = np.fromfile(filepath, dtype=np.uint16, offset=OFFSET).reshape(CT_SHAPE)
    data = data.astype(np.float64) / (2**16 - 1)

    return A63(U, I, tau, series_name, filter_, data)


def load_A63(folderpaths: list[Path]) -> list[A63]:
    """
    Load all A6.3x datasets from given folder paths.
    """
    datasets = []
    for folderpath in folderpaths:
        for filepath in folderpath.glob("*.raw"):
            dataset = load_A63x(filepath)
            datasets.append(dataset)
    return datasets


def load_A64x(filepath: Path) -> np.ndarray:
    """
    Load A6.4x dataset from given file path.
    """
    # Load data and make relative to 0-16bit
    data = np.fromfile(filepath, dtype=np.uint16, offset=OFFSET).reshape(CT_SHAPE)
    data = data.astype(np.float64) / (2**16 - 1)

    return data

def load_A64(folderpaths: list[Path]) -> dict[float, np.ndarray]:
    """
    Load all A6.4x datasets from given folder paths.
    """
    datasets = {}
    for folderpath in folderpaths:
        for filepath in folderpath.glob("*.raw"):
            angle = float(filepath.stem.split("_")[-2].replace("Grad", ""))
            dataset = load_A64x(filepath)
            datasets[angle] = dataset
    return datasets
