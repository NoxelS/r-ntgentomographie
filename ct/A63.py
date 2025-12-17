from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib.transforms import Bbox

from ct.data import load_A63
from ct.plot_config import set_plot_config
from astropy.units.quantity import Quantity


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
        "seaborn-v0_8",
        "seaborn-v0_8-bright",
        "seaborn-v0_8-colorblind",
        "seaborn-v0_8-dark",
        "seaborn-v0_8-dark-palette",
        "seaborn-v0_8-darkgrid",
        "seaborn-v0_8-deep",
        "seaborn-v0_8-muted",
        "seaborn-v0_8-notebook",
        "seaborn-v0_8-paper",
        "seaborn-v0_8-pastel",
        "seaborn-v0_8-poster",
        "seaborn-v0_8-talk",
        "seaborn-v0_8-ticks",
        "seaborn-v0_8-white",
        "seaborn-v0_8-whitegrid",
        "tableau-colorblind10",
    ][7]
)


def A63Analysis():
    series = load_A63(Path("measurement-data").glob("A6.3"))

    # Sort so series without filter come first and then sort by shorter series name
    series.sort(key=lambda x: (x.filter, x.series_name))

    # Ensure results dir exists
    Path("results").mkdir(parents=True, exist_ok=True)

    # Create a 2x3 grid: top row images, bottom row intensity profiles
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Remove padding between subplots
    # fig.subplots_adjust(wspace=0.05, hspace=0.05)

    seriesNameMap = {
        "keile": "Keile im CT",
        "rekalibriertkeile": "Keile im CT (Kalibriert)",
    }

    rowAccumulatedIntensitySeries1 = {
        "Ohne Cu-Filter": [],
        "Ohne Cu-Filter (Kalibriert)": [],
        "Mit Cu-Filter": []
    }

    rowAccumulatedIntensitySeries2 = {
        "Ohne Cu-Filter": [],
        "Ohne Cu-Filter (Kalibriert)": [],
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

        # Compute intensity profile (sum across columns -> one value per row)
        intensity_profile_y = np.sum(data, axis=1)  # length == height
        intensity_profile_x = np.sum(data, axis=0)  # length == width
        rows = np.arange(intensity_profile_y.size)
        cols = np.arange(intensity_profile_x.size)

        # Avoid division by zero when normalizing
        intensity_max_y = float(np.max(intensity_profile_y))
        if intensity_max_y <= 0:
            intensity_max_y = 1.0

        intensity_max_x = float(np.max(intensity_profile_x))
        if intensity_max_x <= 0:
            intensity_max_x = 1.0

        # Map intensity values to image x coordinates (0..width)
        x_mapped = intensity_profile_y / intensity_max_y * (width * 0.95)
        y_mapped = intensity_profile_x / intensity_max_x * (height * 0.95)

        # Plot the profile on top of the image (x: mapped intensity, y: row index)
        ax_img.plot(x_mapped, rows, color="red", alpha=0.5, solid_capstyle="butt", linewidth=1.5, label="CIP (row)")
        ax_img.plot(cols, y_mapped, color="orange", alpha=0.5, solid_capstyle="butt", linewidth=1.5, label="CIP (column)")

        # Horizontal lines indicating the ROI rows (rows ~ 800 and 1800)
        ROIy1 = 790
        ROIy2 = 1900
        ax_img.axhline(ROIy1, color="red", linestyle="--", linewidth=1)
        ax_img.axhline(ROIy2, color="red", linestyle="--", linewidth=1)

        # Vertical lines indicating the ROI columns (columns ~ 200 to 300 and 700 to 800)
        ROIx1 = 1075
        ROIx2 = 1175
        ROIxCross1 = (ROIx1 + ROIx2) // 2
        ROIx3 = 1450
        ROIx4 = 1550
        ROIxCross2 = (ROIx3 + ROIx4) // 2
        ax_img.axvline(ROIxCross1, color="orange", linestyle="--", linewidth=1)
        ax_img.axvline(ROIxCross2, color="orange", linestyle="--", linewidth=1)

        # Keep image axes visible and readable
        ax_img.set_xlabel("$x$ (Pixel)")
        ax_img.set_ylabel("$y$ (Pixel)")
        ax_img.set_xlim(0, width)
        ax_img.set_ylim(height, 0)  # keep origin='upper' behaviour for tick labels

        # Add a secondary X axis on top showing the actual intensity units
        # Map image x (0..width) <-> intensity (0..intensity_max)
        def imgx_to_intensity(x):
            return x / (width * 0.95) * intensity_max_y

        def intensity_to_imgx(x):
            return x / intensity_max_y * (width * 0.95)

        secax = ax_img.secondary_xaxis("top", functions=(imgx_to_intensity, intensity_to_imgx))
        secax.set_xlabel("Summierte Relative Zeilenintensität")

        secaxy = ax_img.secondary_yaxis("right", functions=(imgx_to_intensity, intensity_to_imgx))
        secaxy.set_ylabel("Summierte Relative Spaltenintensität")

        # Disable ticks on secondary axes
        # Remove all tick locations and labels on the secondary axes (top and right)
        secax.set_xticks([])
        secax.set_xticklabels([])
        secax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)

        secaxy.set_yticks([])
        secaxy.set_yticklabels([])
        secaxy.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

        # Optional legend on the image
        ax_img.legend(loc="lower right", fontsize="small")

        # Turn off the corresponding bottom axis (we're overlaying on the image)
        ax_prof.axis("off")

        # Calculate the intensities along the ROI cross sections (so in the x1-x2/2 and x3-x4/2 columns)
        roi_center_x1 = (ROIx1 + ROIx2) // 2
        roi_center_x2 = (ROIx3 + ROIx4) // 2
        intensity_profile_roi1 = data[ROIy1:ROIy2, roi_center_x1]
        intensity_profile_roi2 = data[ROIy1:ROIy2, roi_center_x2]


        # Accumulate intensity profiles for later
        if dataset.filter:
            rowAccumulatedIntensitySeries1["Mit Cu-Filter"].append(intensity_profile_roi1)
            rowAccumulatedIntensitySeries2["Mit Cu-Filter"].append(intensity_profile_roi2)
        else:
            if "rekalibriert" in dataset.series_name.lower():
                rowAccumulatedIntensitySeries1["Ohne Cu-Filter (Kalibriert)"].append(intensity_profile_roi1)
                rowAccumulatedIntensitySeries2["Ohne Cu-Filter (Kalibriert)"].append(intensity_profile_roi2)
            else:
                rowAccumulatedIntensitySeries1["Ohne Cu-Filter"].append(intensity_profile_roi1)
                rowAccumulatedIntensitySeries2["Ohne Cu-Filter"].append(intensity_profile_roi2)

    # Remove extra padding around figure panels
    plt.subplots_adjust(left=0.03, right=0.98, top=0.94, bottom=0.06)
    plt.tight_layout()

    # Remove grid
    for ax_row in axes:
        for ax in ax_row:
            ax.grid(False)

    fig.savefig("results/A63_intensity_profiles.pdf", bbox_inches=
        Bbox.from_bounds(0, 4, 15, 6)
    , pad_inches=0.02)

    # Plot the two accumulated ROI series side-by-side
    from scipy.optimize import curve_fit

    # two side-by-side axes with no horizontal padding so they align exactly
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6), sharey=True,
        gridspec_kw={"wspace": 0.0, "left": 0.03, "right": 0.98}
    )
    names = ["Graphit", "Aluminium"]
    rois = [[1075, 1175], [1450, 1550]]
    all_series = [rowAccumulatedIntensitySeries1, rowAccumulatedIntensitySeries2]

    def exp_decay(x, a, b):
        return a * np.exp(-b * x)

    def b_to_density_b(b, err_b, m, v):
        # Massenschwächungskoeffizenten beta = b / rho
        b_unit = b * u.cm**-1
        err_b_unit = err_b * u.cm**-1

        err_m = 0.001 * u.g
        err_v = np.sqrt(3) * v * (0.25 / 50) # Größtfehler Volumenmessung

        rho = m / v  # g/cm^3

        rho = rho.to(u.g / u.cm**3)
        print(f"Density rho: {rho}")

        # Größfehler
        err_rho = err_m / v + err_v * m / (v**2)

        beta = (b_unit / rho).to(u.cm**2 / u.g)

        # Fehlerfortpflanzung
        err_beta = err_b_unit / rho + err_rho * b_unit / (rho**2)

        # Umrechnung von beta in Dichte
        return beta.to(u.cm**2 / u.g), err_beta.to(u.cm**2 / u.g)

    for i, series in enumerate(all_series):
        ax = axes[i]

        # Left is graphite, right is aluminum alloy
        mass = [20.178 * u.g, 21.410 * u.g][i]
        volume = [1/2 * 50 * 50 * 10 * u.mm**3, 1/2 * 49.5 * 29.5 * 10 * u.mm**3][i]

        roi = 790, 1900
        ax.set_title(f"1D-Intensitätsprofil ({names[i]})")
        ax.set_xlabel("$y$ (cm)")
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

            y_data = accumulated_profile

            ax.plot(x_data, y_data, label=series_name, linewidth=1, color=series_colors.get(series_name, "black"))

            try:
                popt, cov = curve_fit(exp_decay, x_data, y_data, maxfev=10000)
                print(f"Fitted parameters for {series_name} ({names[i]}): {popt}")
                # Errors
                perr = np.sqrt(np.diag(cov))
                print(f"Fitting errors: {perr}")
                beta, beta_err = b_to_density_b(popt[1], perr[1], mass, volume)
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
        ax.set_xlim(left=0, right=4.6)
        ax.legend(fontsize="small")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    fig.savefig("results/A63_accumulated_row_profiles_side_by_side.pdf")

    # Also save each series as a separate figure (one per material)
    for i, series in enumerate(all_series):
        fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 6))
        mass = [20.178 * u.g, 21.410 * u.g][i]
        volume = [1/2 * 50 * 50 * 10 * u.mm**3, 1/2 * 49.5 * 29.5 * 10 * u.mm**3][i]

        ax_single.set_title(f"1D-Intensitätsprofil ({names[i]})")
        ax_single.set_xlabel("$y$ (cm)")
        if i == 0:
            ax_single.set_ylabel("Relative Intensität")

        for series_name, profiles in series.items():
            if len(profiles) == 0:
                continue

            accumulated_profile = np.sum(profiles, axis=0)
            x_data = np.arange(accumulated_profile.size)
            voxelsize = 39.652 * 10**(-4) # in cm
            x_data = x_data * voxelsize
            y_data = accumulated_profile

            ax_single.plot(x_data, y_data, label=series_name, linewidth=1, color=series_colors.get(series_name, "black"))

            try:
                popt, cov = curve_fit(exp_decay, x_data, y_data, maxfev=10000)
                perr = np.sqrt(np.diag(cov))
                beta, beta_err = b_to_density_b(popt[1], perr[1], mass, volume)
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
        ax_single.set_xlim(left=0, right=4.6)
        ax_single.legend(fontsize="small")

        plt.tight_layout()
        filename = f"results/A63_accumulated_row_profiles_{names[i].lower().replace(' ', '_')}.pdf"
        fig_single.savefig(filename)
        plt.close(fig_single)

if __name__ == "__main__":
    A63Analysis()
