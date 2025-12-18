from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.units.quantity import Quantity
from matplotlib.transforms import Bbox

from ct.data import load_A63
from ct.plot_config import set_plot_config


def format_uncertainty(value: Quantity, err: Quantity, sig=2):
    # remove units for formatting; ensure consistency
    v = value.value
    e = err.value

    # round uncertainty to desired significant digits
    e_sig = float(f"{e:.{sig}g}")

    # determine number of decimal places needed
    decimals = max(0, -int(np.floor(np.log10(e_sig))) + (sig - 1))

    # round value accordingly
    v_rounded = round(v, decimals)
    e_rounded = int(round(e_sig * 10**decimals))

    return f"{v_rounded:.{decimals}f}({e_rounded})"

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
    ][7]
)


def A63Analysis():
    series = load_A63(Path("measurement-data").glob("A6.3"))

    # Sort so series without filter come first and then sort by shorter series name
    series.sort(key=lambda x: (x.filter, x.series_name))

    # Ensure results dir exists
    Path("results").mkdir(parents=True, exist_ok=True)

    # Create a 2x2 grid: top row images, bottom row intensity profiles
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False)

    # Remove series that is not calibrated
    series = [s for s in series if s.series_name != "unkalibriert"]

    seriesNameMap = {
        "keile": "Keile im CT",
        "rekalibriertkeile": "Keile im CT",
    }

    rowAccumulatedIntensitySeries1 = {
        "Ohne Cu-Filter": [],
        "Mit Cu-Filter": []
    }

    rowAccumulatedIntensitySeries2 = {
        "Ohne Cu-Filter": [],
        "Mit Cu-Filter": []
    }

    # Overlay profiles on top of images (hide bottom row)
    for i, dataset in enumerate(series):
        ax_img = axes[0, i]
        ax_prof = axes[1, i]  # we'll hide this row later

        data = dataset.data
        height, width = data.shape

        # Show image with origin='upper' so row 0 is at the top
        im = ax_img.imshow(data, cmap="gray", vmin=0, vmax=1, origin="upper", aspect="auto")
        ax_img.set_title(seriesNameMap.get(dataset.series_name.lower(), dataset.series_name) +
                         " (" + ("Mit Cu-Filter" if dataset.filter else "Ohne Cu-Filter") + ")")

        # Turn off the corresponding bottom axis (we're overlaying directly on the image)
        ax_prof.axis("off")

        # Horizontal and vertical ROI definitions
        ROIy1 = 790
        ROIy2 = 1900
        ROIx1 = 1075
        ROIx2 = 1175
        ROIxCross1 = (ROIx1 + ROIx2) // 2
        ROIx3 = 1450
        ROIx4 = 1550
        ROIxCross2 = (ROIx3 + ROIx4) // 2

        from matplotlib.patches import Rectangle

        # Visualize the ROI band used for averaging (±50 px around ROI center)
        band_half_width = 50

        # Rectangle for ROI band 1
        x1_lo = max(0, ROIxCross1 - band_half_width)
        x1_hi = min(width, ROIxCross1 + band_half_width + 1)
        rect1 = Rectangle(
            (x1_lo, ROIy1),
            x1_hi - x1_lo,
            ROIy2 - ROIy1,
            fill=False,
            edgecolor="orange",
            linewidth=1.5,
            linestyle="--",
            label="ROI-Band"
        )
        ax_img.add_patch(rect1)

        # Rectangle for ROI band 2
        x2_lo = max(0, ROIxCross2 - band_half_width)
        x2_hi = min(width, ROIxCross2 + band_half_width + 1)
        rect2 = Rectangle(
            (x2_lo, ROIy1),
            x2_hi - x2_lo,
            ROIy2 - ROIy1,
            fill=False,
            edgecolor="orange",
            linewidth=1.5,
            linestyle="--"
        )
        ax_img.add_patch(rect2)

        # Also outline the full ROI x-ranges for context (optional, thin)
        rect_full1 = Rectangle((ROIx1, ROIy1), ROIx2 - ROIx1, ROIy2 - ROIy1,
                               fill=False, edgecolor="red", linewidth=1.0, linestyle=":")
        rect_full2 = Rectangle((ROIx3, ROIy1), ROIx4 - ROIx3, ROIy2 - ROIy1,
                               fill=False, edgecolor="red", linewidth=1.0, linestyle=":")
        ax_img.add_patch(rect_full1)
        ax_img.add_patch(rect_full2)

        # Keep image axes visible and readable
        ax_img.set_xlabel("$x$ (Pixel)")
        ax_img.set_ylabel("$y$ (Pixel)")
        ax_img.set_xlim(0, width)
        ax_img.set_ylim(height, 0)  # keep origin='upper' behaviour for tick labels

        # Add legend for ROI band
        ax_img.legend(loc="upper right", fontsize="small")

        # Calculate the intensities along the ROI cross sections.
        # Instead of a single column, average over a ±50 px band around the ROI center
        # to reduce noise and pixel-level artifacts.
        roi_center_x1 = (ROIx1 + ROIx2) // 2
        roi_center_x2 = (ROIx3 + ROIx4) // 2

        band_half_width = 50
        x1_lo = max(0, roi_center_x1 - band_half_width)
        x1_hi = min(width, roi_center_x1 + band_half_width + 1)
        x2_lo = max(0, roi_center_x2 - band_half_width)
        x2_hi = min(width, roi_center_x2 + band_half_width + 1)

        # Mean over columns within the band -> one value per row
        intensity_profile_roi1 = np.mean(data[ROIy1:ROIy2, x1_lo:x1_hi], axis=1)
        intensity_profile_roi2 = np.mean(data[ROIy1:ROIy2, x2_lo:x2_hi], axis=1)


        # Accumulate intensity profiles for later
        if dataset.filter:
            rowAccumulatedIntensitySeries1["Mit Cu-Filter"].append(intensity_profile_roi1)
            rowAccumulatedIntensitySeries2["Mit Cu-Filter"].append(intensity_profile_roi2)
        else:
            if "rekalibriert" in dataset.series_name.lower():
                rowAccumulatedIntensitySeries1["Ohne Cu-Filter"].append(intensity_profile_roi1)
                rowAccumulatedIntensitySeries2["Ohne Cu-Filter"].append(intensity_profile_roi2)
            else:
                # As instructed we ignore uncalibrated series for the accumulated profiles
                pass

    # Remove extra padding around figure panels
    plt.subplots_adjust(left=0.03, right=0.98, top=0.94, bottom=0.06)
    plt.tight_layout()

    # Remove grid
    for ax_row in axes:
        for ax in ax_row:
            ax.grid(False)

    fig.savefig("results/A63_intensity_profiles.pdf", bbox_inches=
        Bbox.from_bounds(0, 4, 10, 6)
    , pad_inches=0.02)

    # Plot the two accumulated ROI series side-by-side
    from scipy.optimize import curve_fit

    # two side-by-side axes with no horizontal padding so they align exactly
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6), sharey=True,
        gridspec_kw={"wspace": 0.0, "left": 0.03, "right": 0.98}
    )
    names = ["Graphit", "Aluminium"]
    all_series = [rowAccumulatedIntensitySeries1, rowAccumulatedIntensitySeries2]

    # Wedge dimensions (a, b, h) in mm
    wedge_dims_mm = np.array([
        [50.0, 50.0, 10.0],
        [49.5, 29.5, 10.0],
    ])

    def exp_decay(x, a, b):
        return a * np.exp(-b * x)

    def b_to_density_b(b_fit, err_b_fit, m, a_dim, b_dim, h_dim):
        """
        Convert exponential attenuation parameter b to mass attenuation coefficient.
        """
        # Fit parameter (per thickness) has unit 1/cm
        b_unit = b_fit * u.cm**-1
        err_b_unit = err_b_fit * u.cm**-1

        # Mass uncertainty (scale) as before
        err_m = 0.001 * u.g

        # Volume from explicit dimensions
        V = (0.5 * a_dim * b_dim * h_dim).to(u.cm**3)

        # Volume uncertainty from side-length uncertainties
        delta_L = 0.25 * u.mm  # absolute uncertainty per measured side
        rel_u_V = np.sqrt(
            (delta_L / a_dim) ** 2 +
            (delta_L / b_dim) ** 2 +
            (delta_L / h_dim) ** 2
        )
        err_V = (rel_u_V * V).to(u.cm**3)

        rho = (m / V).to(u.g / u.cm**3)

        # Density uncertainty (first-order propagation)
        err_rho = (err_m / V + err_V * m / (V ** 2)).to(u.g / u.cm**3)

        print(format_uncertainty(rho, err_rho))

        beta = (b_unit / rho).to(u.cm**2 / u.g)

        # Uncertainty propagation for beta = b / rho
        err_beta = (err_b_unit / rho + err_rho * b_unit / (rho ** 2)).to(u.cm**2 / u.g)

        return beta, err_beta

    for i, series in enumerate(all_series):
        ax = axes[i]

        # Left is graphite, right is aluminum alloy
        mass = [20.178 * u.g, 21.410 * u.g][i]
        a_dim, b_dim, h_dim = (wedge_dims_mm[i] * u.mm)

        ax.set_title(f"1D-Intensitätsprofil ({names[i]})")
        ax.set_xlabel("Relative Keildicke (cm)")
        if i == 0:
            ax.set_ylabel("Relative Intensität")

        series_colors = {
            "Ohne Cu-Filter": "blue",
            "Ohne Cu-Filter (Kalibriert)": "orange",
            "Mit Cu-Filter": "green"
        }

        for series_name, profiles in series.items():
            if len(profiles) == 0:
                continue

            accumulated_profile = np.sum(profiles, axis=0)
            x_data = np.arange(accumulated_profile.size)

            # Transform pixel to length in cm relative to start of roi
            voxelsize = 39.652 * 10**(-4) # in cm
            x_data = x_data * voxelsize

            # Transform height values to thickness
            # x_data currently: position along wedge height (cm), measured from start of ROI
            H = 5.0  # cm (50 mm wedge height along detector direction)

            if names[i] == "Graphit":
                Dmax = 5.0  # cm (50 mm depth -> thickness range)
            elif names[i] == "Aluminium":
                Dmax = 2.5  # cm (25 mm depth -> thickness range)
            else:
                raise ValueError("Unknown material for thickness conversion")

            # Convert height-coordinate to physical thickness in beam direction
            x_data = (Dmax / H) * x_data

            y_data = accumulated_profile

            ax.plot(x_data, y_data, label=series_name, linewidth=1, color=series_colors.get(series_name, "black"))

            try:
                popt, cov = curve_fit(exp_decay, x_data, y_data, maxfev=10000)
                print(f"Fitted parameters for {series_name} ({names[i]}): {popt}")
                # Errors
                perr = np.sqrt(np.diag(cov))
                print(f"Fitting errors: {perr}")
                beta, beta_err = b_to_density_b(popt[1], perr[1], mass, a_dim, b_dim, h_dim)
                label = f"{series_name} Exp-Fit $a e^{{-\\frac{{\\mu y}}{{\\rho}}}}$: $\\mu = {format_uncertainty(beta, beta_err)} \\ cm^2g^{{-1}}$"
                fitted_curve = exp_decay(x_data, *popt)
                ax.plot(x_data, fitted_curve, linestyle="--", label=label, linewidth=1, color=series_colors.get(series_name, "black"))
            except Exception as e:
                print(f"Could not fit exponential decay for {series_name} ({names[i]})")
                print(e)

        # Set x ticks style
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

        ax.set_yscale("log")
        ax.set_yticks([0.01, 0.1, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.01%", "10%", "25%", "50%", "75%", "100%"])
        ax.set_ylim(0.01, 1.0)
        ax.set_xlim(left=0)
        ax.legend(fontsize="small")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    fig.savefig("results/A63_accumulated_row_profiles_side_by_side.pdf")

    # Also save each series as a separate figure (one per material)
    for i, series in enumerate(all_series):
        fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 6))
        mass = [20.178 * u.g, 21.410 * u.g][i]
        a_dim, b_dim, h_dim = (wedge_dims_mm[i] * u.mm)

        ax_single.set_title(f"1D-Intensitätsprofil ({names[i]})")
        ax_single.set_xlabel("$Relative Keildicke$ (cm)")
        if i == 0:
            ax_single.set_ylabel("Relative Intensität")

        for series_name, profiles in series.items():
            if len(profiles) == 0:
                continue

            accumulated_profile = np.sum(profiles, axis=0)
            x_data = np.arange(accumulated_profile.size)
            voxelsize = 39.652 * 10**(-4) # in cm
            x_data = x_data * voxelsize

            # Transform height values to thickness
            # x_data currently: position along wedge height (cm), measured from start of ROI
            H = 5.0  # cm (50 mm wedge height along detector direction)
            if names[i] == "Graphit":
                Dmax = 5.0  # cm (50 mm depth -> thickness range)
            elif names[i] == "Aluminium":
                Dmax = 2.5  # cm (25 mm depth -> thickness range)
            else:
                raise ValueError("Unknown material for thickness conversion")

            # Convert height-coordinate to physical thickness in beam direction
            x_data = (Dmax / H) * x_data

            y_data = accumulated_profile

            ax_single.plot(x_data, y_data, label=series_name, linewidth=1, color=series_colors.get(series_name, "black"))

            try:
                popt, cov = curve_fit(exp_decay, x_data, y_data, maxfev=10000)
                perr = np.sqrt(np.diag(cov))
                beta, beta_err = b_to_density_b(popt[1], perr[1], mass, a_dim, b_dim, h_dim)
                label = f"{series_name} Exp-Fit $a e^{{-\\frac{{\\mu y}}{{\\rho}}}}$: $\\mu = {format_uncertainty(beta, beta_err)} \\ cm^2g^{{-1}}$"
                fitted_curve = exp_decay(x_data, *popt)
                ax_single.plot(x_data, fitted_curve, linestyle="--", label=label, linewidth=1, color=series_colors.get(series_name, "black"))
            except Exception as e:
                print(f"Could not fit exponential decay for {series_name} ({names[i]})")
                print(e)

        ax_single.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax_single.set_yscale("log")
        ax_single.set_yticks([0.01, 0.1, 0.25, 0.5, 0.75, 1.0])
        ax_single.set_yticklabels(["0.01%", "10%", "25%", "50%", "75%", "100%"])
        ax_single.set_ylim(0.01, 1.0)
        ax_single.set_xlim(left=0)
        ax_single.legend(fontsize="small")

        plt.tight_layout()
        filename = f"results/A63_accumulated_row_profiles_{names[i].lower().replace(' ', '_')}.pdf"
        fig_single.savefig(filename)
        plt.close(fig_single)

if __name__ == "__main__":
    A63Analysis()
