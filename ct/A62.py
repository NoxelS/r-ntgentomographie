from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.optimize import curve_fit

from ct.data import load_A62
from ct.plot_config import set_plot_config

# Set plot config
set_plot_config(
    [
        "Solarize_Light2",
        "_classic_test_patch",
        "_mpl-gallery",
        "_mpl-gallery-nogrid",
        "bmh",
        "classic",
        "dark_background",
        "fast",
        "fivethirtyeight",
        "ggplot",
        "grayscale",
        "petroff10",
        "tableau-colorblind10",
    ][1]
)


def _linear_with_bias(x, a, b):
    return a * x + b


def _linear_no_bias(x, a):
    return a * x


def fit_linear_models(x, y, yerr=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    sigma = None
    absolute_sigma = False
    if yerr is not None:
        sigma = np.asarray(yerr, dtype=float)
        absolute_sigma = True

    popt0, pcov0 = curve_fit(
        _linear_no_bias, x, y,
        sigma=sigma, absolute_sigma=absolute_sigma,
        maxfev=10000,
    )
    perr0 = np.sqrt(np.diag(pcov0))

    popt1, pcov1 = curve_fit(
        _linear_with_bias, x, y,
        sigma=sigma, absolute_sigma=absolute_sigma,
        maxfev=10000,
    )
    perr1 = np.sqrt(np.diag(pcov1))

    return {
        "no_bias": {"popt": popt0, "perr": perr0},
        "with_bias": {"popt": popt1, "perr": perr1},
    }


def annotate_fit(ax, x, fit, label_prefix=""):
    """Overlay fit curves and provide compact legend labels."""
    x = np.asarray(x, dtype=float)
    xs = np.linspace(0, np.max(x), 200)

    a0 = fit["no_bias"]["popt"][0]
    da0 = fit["no_bias"]["perr"][0]
    ax.plot(
        xs,
        _linear_no_bias(xs, a0),
        linestyle="--",
        linewidth=1.5,
        label=f"{label_prefix}Fit ohne Bias: a={a0:.4g}±{da0:.2g}",
    )

    a1, b1 = fit["with_bias"]["popt"]
    da1, db1 = fit["with_bias"]["perr"]
    ax.plot(
        xs,
        _linear_with_bias(xs, a1, b1),
        linestyle=":",
        linewidth=1.2,
        label=f"{label_prefix}Fit mit Bias: a={a1:.4g}±{da1:.2g}, b={b1:.4g}±{db1:.2g}",
    )


def A62Analysis():
    # Uncertainties in U and I
    dU = 0.1 * u.kV
    dI = 0.3 * u.uA
    dTau = 1 * u.ms

    Path("results").mkdir(parents=True, exist_ok=True)

    datasets = load_A62(Path("measurement-data").glob("A6.2.*"))

    series_names = {
        "grauwerte": r"6.2.1 ($\delta I$)",
        "nach-neukalibrierung-grauwerte": r"6.2.1nk ($\delta I$)",
        "GrauwertCheck": r"6.2.1x ($\delta I$)",
        "integrationszeit":  r"6.2.2 ($\delta \tau$)",
        "IntegrationszeitCheck": r"6.2.2x ($\delta \tau$)",
        "Anodenspannung": r"6.2.3 ($\delta U$)",
    }

    series_colors = [
        "tab:orange",
        "tab:red",
        "tab:red",
        "tab:cyan",
        "tab:purple",
        "tab:blue",
    ]

    measurement_series = {}

    # Calculate J and its uncertainty for each dataset
    for dataset in datasets:
        series_name = series_names.get(dataset.series_name)
        U = dataset.U
        I = dataset.I
        tau = dataset.tau
        mean, variance, std = dataset.stats

        # Calculate integrated intensity J
        J = U**2 * I * tau
        J.to(u.V**2 * u.A * u.s)

        err_J = np.abs(2 * dU * U * I * tau) + np.abs(U**2 * dI * tau) + np.abs(U**2 * I * (dTau))

        if series_name not in measurement_series:
            measurement_series[series_name] = []

        measurement_series[series_name].append((dataset, J, err_J, mean, std))

    regression_results = []

    max_J_value = max(
        max(item[1].value for item in series)
        for series in measurement_series.values()
    )

    for i, series_name in enumerate(measurement_series):
        if series_name == "6.2.1nk ($\\delta I$)":
            continue  # Skip this series for now

        series = measurement_series[series_name]
        color = series_colors[i]

        Js = np.array([item[1].value for item in series], dtype=float)
        err_Js = np.array([item[2].value for item in series], dtype=float)
        Is = np.array([item[3] for item in series], dtype=float)
        Is_err = np.array([item[4] for item in series], dtype=float)

        # Fit: primary without bias; biased fit only for diagnostic comparison.
        fit = fit_linear_models(Js, Is, yerr=Is_err)

        # Store results (both variants) for the summary table
        a0 = fit["no_bias"]["popt"][0]
        da0 = fit["no_bias"]["perr"][0]
        a1, b1 = fit["with_bias"]["popt"]
        da1, db1 = fit["with_bias"]["perr"]

        regression_results.append(
            {
                "series": series_name,
                "slope_no_bias": a0,
                "slope_no_bias_err": da0,
                "slope_bias": a1,
                "slope_bias_err": da1,
                "intercept": b1,
                "intercept_err": db1,
            }
        )

        # ---- Separate plot per series ----
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(r"Integrierte Relative CT Intensität: " + series_name)
        ax.set_xlabel(r"$U^2 \cdot I \cdot \tau\ (\mathrm{V^2As})$")
        ax.set_ylabel(r"Integrierte Mittlere Relative Intensität $I_{\mathrm{mean}}$")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # y ticks in percent
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        # x ticks in scientific notation (kilo scaling)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fr"${x / 1e3:.0f} \cdot 10^{{3}}$"))

        ax.errorbar(
            Js,
            Is,
            xerr=err_Js,
            yerr=Is_err,
            fmt="x",
            color=color,
            markersize=4,
            linestyle="None",
            capsize=2,
            elinewidth=1,
            label="Messdaten",
        )

        annotate_fit(ax, Js, fit)

        ax.set_xlim(left=0, right=max_J_value)
        ax.set_ylim(bottom=0)
        ax.legend()
        fig.tight_layout()

        safe_name = (
            series_name
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("$", "")
            .replace("\\", "")
            .replace("{", "")
            .replace("}", "")
            .replace(":", "")
            .replace("/", "_")
        )
        fig.savefig(f"results/A62_fit_{safe_name}.pdf")
        plt.close(fig)

    # ---- Compact comparison plot (no-bias fits only) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(r"Kramers-Skalierung: Vergleich (Fits ohne Bias)")
    ax.set_xlabel(r"$U^2 \cdot I \cdot \tau\ (\mathrm{V^2As})$")
    ax.set_ylabel(r"Integrierte Mittlere Relative Intensität $I_{\mathrm{mean}}$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fr"${x / 1e3:.0f} \cdot 10^{{3}}$"))

    xs = np.linspace(0, max_J_value, 300)
    for i, res in enumerate(regression_results):
        # Map back to color by matching series order
        sname = res["series"]
        # Find original color
        color = series_colors[list(measurement_series.keys()).index(sname)]
        ax.plot(xs, _linear_no_bias(xs, res["slope_no_bias"]), linewidth=1.2, color=color, label=sname)

    ax.set_xlim(left=0, right=max_J_value)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/A62_fits_no_bias_comparison.pdf")
    plt.close(fig)

    # Plot regression results in a matplotlib table
    fig, ax = plt.subplots(figsize=(10, len(regression_results) * 0.5 + 1))
    ax.axis("off")

    table_data = [
        [
            result["series"],
            f"$({result['slope_no_bias'] * 1e6:.2f} \\pm {result['slope_no_bias_err'] * 1e6:.2f})\\cdot 10^{{-6}}$",
            f"$({result['slope_bias'] * 1e6:.2f} \\pm {result['slope_bias_err'] * 1e6:.2f})\\cdot 10^{{-6}}$",
            f"$({result['intercept']*100:.2f} \\pm {result['intercept_err']*100:.2f})$",
        ]
        for result in regression_results
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["Messreihe", r"Steigung ohne Bias ($V^{-2}A^{-1}s^{-1}$)", r"Steigung mit Bias ($V^{-2}A^{-1}s^{-1}$)", "Offset (%)"],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    plt.tight_layout()

    # Remove padding of table
    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)

    plt.title("A6.2: Fitparameter (primär ohne Bias, mit Bias nur Diagnose)")
    plt.savefig("results/A62_regression_results.pdf")


if __name__ == "__main__":
    A62Analysis()
