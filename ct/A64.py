from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ct.data import load_A64
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


def A64Analysis():
    import matplotlib.gridspec as gridspec

    series = load_A64(Path("measurement-data").glob("A6.4"))

    # sort angles numerically so first is 0 and last is 360 (if present)
    try:
        angles = sorted(series.keys(), key=lambda a: float(a))
    except Exception:
        angles = sorted(series.keys())

    # Grid: 3 image rows each followed by a small trace row -> 6 rows x 3 cols
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(6, 3, height_ratios=[4, 1, 4, 1, 4, 1], hspace=0.08, wspace=0.12)

    # original horizontal line (in original image coordinates)
    y_line = 1650

    # crop amounts: top, right, bottom, left
    a, b, c, d = 1250, 400, 600, 400

    img_axes = []
    im = None

    # keep pairs so we can align trace axes to image axes after any layout changes
    axis_pairs = []

    # Intensity spectrums for each angle
    intensity_spectrums = {}

    for idx, angle in enumerate(angles):
        row = idx // 3
        col = idx % 3

        data = series[angle]

        # Safely compute cropping bounds and crop the array
        h, w = data.shape
        top = min(a, max(0, h - 2))
        bottom = min(c, max(0, h - top - 1))
        left = min(d, max(0, w - 2))
        right = min(b, max(0, w - left - 1))
        data = data[top : h - bottom, left : w - right]

        # Adjust y_line and x interval to cropped coordinates
        y_adj = y_line - top
        # original x interval
        x0_orig, x1_orig = 775, 1775
        x0_adj = x0_orig - left
        x1_adj = x1_orig - left
        # Clamp to cropped image bounds
        y = int(max(0, min(y_adj, data.shape[0] - 1)))
        x0 = int(max(0, min(x0_adj, data.shape[1] - 1)))
        x1 = int(max(0, min(x1_adj, data.shape[1])))
        x_interval = (x0, x1)

        # Image axis (on even-numbered rows of the GridSpec)
        ax_img = fig.add_subplot(gs[row * 2, col])
        im = ax_img.imshow(data, cmap="viridis", vmin=0, vmax=1)
        ax_img.axis("off")

        # add title showing theta
        if angle is not None:
            ax_img.set_title(f"$\\theta = {angle} \\degree$", fontsize=10, y=1, color="black")

        # draw horizontal dashed line at adjusted y
        ax_img.hlines(y, *x_interval, colors="red", linewidth=0.75, linestyles="--", alpha=0.7)

        img_axes.append(ax_img)

        # Trace axis directly under the image (odd-numbered rows of the GridSpec)
        ax_trace = fig.add_subplot(gs[row * 2 + 1, col])
        # extract trace within clamped x interval
        if x1 > x0:
            trace = data[y, x0:x1]
            ax_trace.plot(np.arange(trace.size) + x0 + left, trace, color="red", linewidth=0.5)
            # set x-limits back to original-coordinate view (optional)
            ax_trace.set_xlim(x0 + left, x1 + left)
        else:
            # empty trace if interval invalid
            ax_trace.plot([], [])
            ax_trace.set_xlim(0, 1)

        ax_trace.set_xlabel("$x$ (Pixel)", fontsize=8)

        if angle in [0.0, 135.0, 270.0]:
            ax_trace.set_ylabel("Relative Intensität", fontsize=8)

        ax_trace.tick_params(axis="both", which="major", labelsize=6)

        axis_pairs.append((ax_img, ax_trace))

        # Add to intensity spectrums
        intensity_spectrums[angle] = trace

    # leave space on the right for a global colorbar
    fig.subplots_adjust(right=0.88)

    # add a single colorbar for all image axes (pad adjusts distance from subplots)
    if im is not None:
        cbar = fig.colorbar(im, ax=img_axes, shrink=0.6, pad=0.02)

    # After any layout changes (like colorbar), force each trace axis to match the image axis width
    # and sit immediately beneath it. We compute the trace height relative to the image axis
    # using the GridSpec height ratio (images:traces = 4:1) so the trace keeps the intended height.
    gap = 0.005  # small gap between image and trace
    for ax_img, ax_trace in axis_pairs:
        pos_img = ax_img.get_position()  # Bbox in figure coordinates
        # maintain the same horizontal position and width as the image axis
        # trace height is roughly a quarter of the image axis height (since 4:1 ratio)
        trace_height = pos_img.height * (1 / 4)
        new_y0 = pos_img.y0 - trace_height - gap
        # set the new position for the trace axis
        ax_trace.set_position([pos_img.x0, new_y0, pos_img.width, trace_height])
        # ensure x-limits match exactly
        lines = ax_trace.get_lines()
        if lines:
            xdata = lines[0].get_xdata()
            if xdata.size > 0:
                ax_trace.set_xlim(xdata.min(), xdata.max())

    fig.savefig("results/A64_all_images.pdf", bbox_inches="tight")


    # Figure out where the balls are
    ball_positions = np.zeros((4, 4))
    ball_positions[0, 0] = 1
    ball_positions[0, 1] = 1
    ball_positions[1, 0] = 1
    ball_positions[1, 1] = 1
    ball_positions[1, 2] = 1
    ball_positions[1, 3] = 1
    ball_positions[2, 3] = 1
    ball_positions[3, 0] = 1
    ball_positions[3, 2] = 1

    # Visualize the 4x4 ball matrix: filled circle => ball (1), outline => no ball (0)
    fig, ax = plt.subplots(figsize=(4, 4))
    nrows, ncols = ball_positions.shape
    radius = 0.4

    for r in range(nrows):
        for c in range(ncols):
            val = ball_positions[r, c]
            facecolor = "black" if val == 1 else "none"
            edgecolor = "black"
            circ = plt.Circle((c, r), radius=radius, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
            ax.add_patch(circ)

    # Layout so matrix indices align visually like an image (row 0 at top)
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticklabels(range(1, ncols + 1))
    ax.set_yticklabels(range(1, nrows + 1))
    ax.set_xlabel("Spalte")
    ax.set_ylabel("Zeile")
    ax.set_title("Kugelpositionen", fontsize=10)
    plt.tight_layout()
    fig.savefig("results/A64_ball_positions.pdf", bbox_inches="tight")
    plt.close(fig)


    # Analyse the spectrums in 3d using all available angles
    # Collect available angles from the intensity_spectrums dict and sort numerically if possible
    available_angles = list(intensity_spectrums.keys())
    try:
        target_angles = sorted(available_angles, key=lambda a: float(a))
    except Exception:
        target_angles = sorted(available_angles)

    # Determine ROI size from the first available spectrum
    first_spec = None
    for k in target_angles:
        s = intensity_spectrums.get(k)
        if s is not None and s.size > 0:
            first_spec = s
            break
    if first_spec is None:
        raise RuntimeError("No valid intensity spectrums found to build probability field.")

    ROI_size = first_spec.shape[0], first_spec.shape[0]
    print("Intensity spectrums ROI size:", ROI_size)

    probabilityField = np.zeros((ROI_size[0], ROI_size[1], len(target_angles)), dtype=float)

    # helper: crop or pad a 2D array to target shape (centered)
    def center_crop_or_pad(arr, target_shape):
        out = np.zeros(target_shape, dtype=arr.dtype)
        h0, w0 = arr.shape
        ht, wt = target_shape
        # compute cropping/padding ranges
        start_h = max(0, (ht - h0) // 2)
        start_w = max(0, (wt - w0) // 2)
        src_start_h = max(0, (h0 - ht) // 2)
        src_start_w = max(0, (w0 - wt) // 2)
        copy_h = min(h0, ht)
        copy_w = min(w0, wt)
        out[start_h : start_h + copy_h, start_w : start_w + copy_w] = arr[
            src_start_h : src_start_h + copy_h, src_start_w : src_start_w + copy_w
        ]
        return out

    # try to use scipy.ndimage.rotate for arbitrary-angle rotation if available
    try:
        from scipy import ndimage as ndi  # type: ignore

        _have_ndi = True
    except Exception:
        _have_ndi = False

    for angle_index, angle in enumerate(target_angles):
        spectrum = intensity_spectrums.get(angle)
        if spectrum is None or spectrum.size == 0:
            # leave zero slice if no spectrum
            continue

        # Ensure 1D
        spectrum = np.asarray(spectrum).ravel()

        # Make a 2D tiled surface from the 1D spectrum (rows = spectrum positions)
        spectrum_3d = np.tile(spectrum, (ROI_size[1], 1)).T  # shape (ROI_size[0], ROI_size[1])

        # Rotate according to the angle (if possible). If scipy not available, fall back to nearest 90° rot.
        try:
            angle_float = float(angle)
        except Exception:
            angle_float = 0.0

        if _have_ndi:
            # ndi.rotate rotates counter-clockwise by default. keep shape by reshape=False.
            rotated_spectrum = ndi.rotate(spectrum_3d, angle_float, reshape=False, order=1, mode="nearest")
        else:
            # fallback: rotate to nearest multiple of 90 degrees
            k = int(np.round(angle_float / 90.0)) % 4
            # np.rot90 rotates 90 degrees counter-clockwise k times
            rotated_spectrum = np.rot90(spectrum_3d, k=k)

        # Ensure final slice matches ROI_size (crop or pad if necessary)
        if rotated_spectrum.shape != ROI_size:
            rotated_spectrum = center_crop_or_pad(rotated_spectrum, ROI_size)

        probabilityField[:, :, angle_index] = rotated_spectrum

    # Plot the 3d probability field
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(ROI_size[1]), np.arange(ROI_size[0]))

    # Accumulate the intensity values across angles
    Z = np.sum(probabilityField, axis=2)

    # Normalize Z for better visualization
    Z = Z / np.max(Z) if np.max(Z) != 0 else Z

    # Invert Z
    Z = 1 - Z

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Intensität')

    # Plot the real ball positions on top of the surface
    ball_x, ball_y = np.where(ball_positions == 1)

    # Map to pixel coordinates
    ball_x = ball_x * (ROI_size[0] // ball_positions.shape[0]) + (ROI_size[0] // (2 * ball_positions.shape[0]))
    ball_y = ball_y * (ROI_size[1] // ball_positions.shape[1]) + (ROI_size[1] // (2 * ball_positions.shape[1]))


    # Add offset in x, y so the balls align better with the surface peaks
    scale = 0.775
    offset = [95, 120]
    ball_x = ball_x * scale + offset[0]
    ball_y = ball_y * scale + offset[1]
    ball_z = Z[ball_x.astype(int), ball_y.astype(int)] + 0.02  # slight offset above surface
    ax.scatter(ball_y, ball_x, zs=ball_z, zdir='z', color='red', s=100, label='Correct Ball Positions')

    # Hide grid
    ax.grid(False)

    # Rotate by 45 degrees for better view
    ax.view_init(elev=50, azim=120)

    ax.set_xlabel('$x$ (Pixel)')
    ax.set_ylabel('$y$ (Pixel)')
    ax.set_zlabel('Itensität')
    ax.set_title('3D Wahrscheinlichkeitsdichte der Kugelpositionen')
    plt.legend()
    fig.savefig("results/A64_probability_field_3D.pdf", bbox_inches="tight")

    # Plot this probability field as 2D heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(Z, cmap='viridis', origin='lower')
    fig.colorbar(heatmap, ax=ax, label='Intensität')

    # Plot the real ball positions on top of the heatmap
    ax.scatter(ball_y, ball_x, color='red', s=100, label='Correct Ball Positions')
    ax.legend()

    ax.set_xlabel('$x$ (Pixel)')
    ax.set_ylabel('$y$ (Pixel)')
    ax.set_title('2D Wahrscheinlichkeitsdichte der Kugelpositionen')
    fig.savefig("results/A64_probability_field_2D.pdf", bbox_inches="tight")


if __name__ == "__main__":
    A64Analysis()
