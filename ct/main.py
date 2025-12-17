from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from ct.plot_config import set_plot_config

CT_SHAPE = (2464, 2976)

# Set plot config
set_plot_config(['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'petroff10', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10'][1])

def ct_histogram_raw(filepath: str, shape: tuple[int, int], dtype=np.uint16):
    """
    Reads a raw computed tomography image and computes its histogram.

    Parameters
    ----------
    filepath : str
        Path to the raw image file.
    shape : tuple[int, int]
        Image dimensions (height, width).
    dtype : numpy dtype, optional
        Data type of the pixels (default: uint16, typical for CT).
    plot : bool, optional
        Whether to display the histogram plot.

    Returns
    -------
    hist : np.ndarray
        Histogram counts.
    bin_edges : np.ndarray
        Histogram bin edges.
    image : np.ndarray
        2D image array.
    """
    # Read raw binary data
    with open(filepath, "rb") as f:
        image = np.fromfile(f, dtype=dtype , offset=2048).reshape(shape)
        # Make relative to 0-16bit
        image = image.astype(np.float64) / (2**16 - 1)

    # Compute mean
    mean_val = np.mean(image)
    variance_val = np.var(image)
    std = np.std(image)

    return mean_val, variance_val, std



if __name__ == "__main__":
    # Folder with current change data
    data_I_path = "data/A6.2.1/"

    series_1 = [] # dt=0.15s
    series_2 = [] # dt=0.2s
    series_3 = [] # Integrationszeit 5μA
    series_4 = [] # Integrationszeit 10μA
    series_5 = [] # I = 70ua
    series_6 = [] # I = 110ua
    series_1_check = [] # dt=0.15s recalibrated
    series_2_check = [] # dt=0.2s recalibrated

    series_1_data = {"U": 100, "dt": 0.15}
    series_2_data = {"U": 100, "dt": 0.2}

    for file in Path(data_I_path).glob("*"):
        # Read U,i, dt from filename but only read in numbers and . for decimals
        _1, U_str, I_str, dt_str, _2 = file.name.split("_")
        U = float(U_str.replace("kV", ""))
        I = float(I_str.replace("muA", "").replace("uA", ""))
        dt = float(dt_str.replace("s", ""))

        J = U**2 * I * dt

        dU = 0.1
        dI = 0.3

        err_J = np.sqrt((2 * dU * U * I)**2 + (U**2 * dI)**2)


        mean, var, std = ct_histogram_raw(str(file), CT_SHAPE, dtype=np.uint16)

        if _1 == "grauwerte":
            if dt == series_1_data["dt"] and series_1_data["U"] == U:
                series_1.append(np.array([J, mean, var, std, err_J]))
            elif dt == series_2_data["dt"] and series_2_data["U"] == U:
                series_2.append(np.array([J, mean, var, std, err_J]))
        if _1 == "integrationszeit":
            if I == 5:
                series_3.append(np.array([J, mean, var, std, err_J]))
            elif I == 10:
                series_4.append(np.array([J, mean, var, std, err_J]))
        if _1 == "Anodenspannung":
            if I == 70:
                series_5.append(np.array([J, mean, var, std, err_J]))
            elif I == 110:
                series_6.append(np.array([J, mean, var, std, err_J]))
        if _1 == "IntegrationsCheck":
            if dt == series_1_data["dt"] and series_1_data["U"] == U:
                series_1_check.append(np.array([J, mean, var, std, err_J]))
            elif dt == series_2_data["dt"] and series_2_data["U"] == U:
                series_2_check.append(np.array([J, mean, var, std, err_J]))

    series_1 = np.array(series_1)
    series_2 = np.array(series_2)
    series_3 = np.array(series_3)
    series_4 = np.array(series_4)
    series_5 = np.array(series_5)
    series_6 = np.array(series_6)
    series_1_check = np.array(series_1_check)
    series_2_check = np.array(series_2_check)

    # Sort series by J
    series_1 = series_1[series_1[:, 0].argsort()]
    series_2 = series_2[series_2[:, 0].argsort()]
    series_3 = series_3[series_3[:, 0].argsort()]
    series_4 = series_4[series_4[:, 0].argsort()]
    series_5 = series_5[series_5[:, 0].argsort()]
    series_6 = series_6[series_6[:, 0].argsort()]
    series_1_check = series_1_check[series_1_check[:, 0].argsort()]
    series_2_check = series_2_check[series_2_check[:, 0].argsort()]

    series_2_point = series_2[-1]
    series_2 = series_2[:-1] # Last point is measured on recalibrated

    # # Calculate linear regressions with scipy
    # reg1 = linregress(series_1[:, 0], series_1[:, 1])
    # reg2 = linregress(series_2[:, 0], series_2[:, 1])
    # reg3 = linregress(series_3[:, 0], series_3[:, 1])
    # reg4 = linregress(series_4[:, 0], series_4[:, 1])

    # print(f"Series 1: slope={reg1.slope}, intercept={reg1.intercept}, r_value={reg1.rvalue}, p_value={reg1.pvalue}, std_err={reg1.stderr}")
    # print(f"Series 2: slope={reg2.slope}, intercept={reg2.intercept}, r_value={reg2.rvalue}, p_value={reg2.pvalue}, std_err={reg2.stderr}")

    # x_range = np.array([0, max(series_1[:, 0].max(), series_2[:, 0].max())])

    # label1 = f"Fit $({reg1.slope*1e6:.2f} \\pm {reg1.stderr*1e6:.2f}) UI + {reg1.intercept:.2f} \\pm {reg1.intercept_stderr:.2f}$"
    # label2 = f"Fit $({reg2.slope*1e6:.2f} \\pm {reg2.stderr*1e6:.2f}) UI + {reg2.intercept:.2f} \\pm {reg2.intercept_stderr:.2f}$"
    # label3 = f"Fit $({reg3.slope*1e6:.2f} \\pm {reg3.stderr*1e6:.2f}) UI + {reg3.intercept:.2f} \\pm {reg3.intercept_stderr:.2f}$"
    # label4 = f"Fit $({reg4.slope*1e6:.2f} \\pm {reg4.stderr*1e6:.2f}) UI + {reg4.intercept:.2f} \\pm {reg4.intercept_stderr:.2f}$"

    # plt.plot(x_range, reg1.slope * x_range + reg1.intercept, label="Fit1", linestyle="--", linewidth=0.5)
    # plt.plot(x_range, reg2.slope * x_range + reg2.intercept, label="Fit2", linestyle="--", linewidth=0.5)
    # plt.plot(x_range, reg3.slope * x_range + reg3.intercept, label="Fit3", linestyle="--", linewidth=0.5)
    # plt.plot(x_range, reg4.slope * x_range + reg4.intercept, label="Fit4", linestyle="--", linewidth=0.5)

    # Plot the results and use std as error bars
    labels = ["dt=0.15s", "dt=0.2s", "I=5μA", "I=10μA", "I=70μA", "I=110μA", "dt=0.15s recalibrated", "dt=0.2s recalibrated"]
    for series, label in zip([series_1, series_2, series_3, series_4, series_5, series_6, series_1_check, series_2_check], labels):
        reg = linregress(series[:, 0], series[:, 1])
        x_range = np.array([0, series[:, 0].max()])
        plt.plot(x_range, reg.slope * x_range + reg.intercept, label=f"Fit {label}", linestyle="--", linewidth=0.5)

        plt.errorbar(
            series[:, 0],
            series[:, 1],
            yerr=series[:, 3],
            xerr=series[:, 4],
            label=label,
            marker="",
            linestyle=":",
            linewidth=0.5,
            capsize=1,
            elinewidth=1,
        )

    plt.xlabel("$U^2 \\cdot I \\cdot \\tau$ (kV$^2$·μA·s)")
    plt.ylabel("Mean relative pixel value")
    plt.title("Mean relative pixel value vs. J for different exposure times")

    # Set origin to 0
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.legend()
    plt.grid()

    plt.show()

