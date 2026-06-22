"""Create a paper-ready visualization of processed Sleep-EDF EEG/EOG epochs.

The script reads the processed arrays without loading the full signal tensor
into memory, selects a small but informative subset, and saves a composite
figure showing class balance, a continuous hypnogram window, representative
30-second multi-channel epochs, and stage-wise spectral fingerprints.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


DEFAULT_DATA_DIR = Path("data/data_multimodal_eeg_eog_3ch")
DEFAULT_OUT_DIR = Path("results/dataset_visualization")
DEFAULT_CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
DEFAULT_CHANNEL_NAMES = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
STAGE_COLORS = ["#f2b84b", "#e76f51", "#4e79a7", "#2a9d8f", "#8e6bd8"]
CHANNEL_COLORS = ["#314f9f", "#00a6a6", "#d45b7a"]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 130,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_class_names(data_dir: Path) -> list[str]:
    path = data_dir / "label_map.csv"
    if not path.exists():
        return DEFAULT_CLASS_NAMES
    df = pd.read_csv(path)
    if {"label", "stage"}.issubset(df.columns):
        return df.sort_values("label")["stage"].astype(str).tolist()
    return DEFAULT_CLASS_NAMES


def load_channel_names(data_dir: Path) -> list[str]:
    path = data_dir / "channel_info.csv"
    if not path.exists():
        return DEFAULT_CHANNEL_NAMES
    df = pd.read_csv(path)
    if {"channel_index", "channel_name"}.issubset(df.columns):
        return df.sort_values("channel_index")["channel_name"].astype(str).tolist()
    return DEFAULT_CHANNEL_NAMES


def short_channel_names(channel_names: list[str]) -> list[str]:
    names = []
    for name in channel_names:
        clean = name.replace("EEG ", "").replace(" horizontal", "")
        names.append(clean)
    return names


def load_arrays(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    x_path = data_dir / "X_all.npy"
    y_path = data_dir / "y_all.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing X_all.npy or y_all.npy in {data_dir}")
    x = np.load(x_path, mmap_mode="r")
    y = np.asarray(np.load(y_path, mmap_mode="r"), dtype=np.int64)
    if x.ndim != 3:
        raise ValueError(f"Expected X_all with shape [N, C, T], got {x.shape}")
    if len(y) != x.shape[0]:
        raise ValueError(f"X/y length mismatch: {x.shape[0]} vs {len(y)}")
    return x, y


def load_epoch_info(data_dir: Path) -> pd.DataFrame | None:
    path = data_dir / "split_epoch_info.csv"
    if not path.exists():
        return None
    keep_cols = ["index", "subject_id", "record_id", "label"]
    try:
        df = pd.read_csv(path, usecols=keep_cols)
    except ValueError:
        df = pd.read_csv(path)
    if "index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "index"})
    return df


def class_counts(labels: np.ndarray, n_classes: int) -> np.ndarray:
    valid = labels[(labels >= 0) & (labels < n_classes)]
    return np.bincount(valid, minlength=n_classes)


def shannon_entropy(values: np.ndarray, n_classes: int) -> float:
    counts = class_counts(values, n_classes).astype(float)
    probs = counts[counts > 0] / max(counts.sum(), 1.0)
    return float(-(probs * np.log2(probs)).sum())


def choose_hypnogram_window(
    labels: np.ndarray,
    n_classes: int,
    window_epochs: int,
    epoch_info: pd.DataFrame | None,
) -> tuple[np.ndarray, str]:
    window_epochs = min(window_epochs, len(labels))
    best_score = -1.0
    best_indices = np.arange(window_epochs)
    best_label = "dataset window"

    if epoch_info is not None and {"index", "record_id", "label"}.issubset(epoch_info.columns):
        for record_id, group in epoch_info.groupby("record_id", sort=False):
            idx = group["index"].to_numpy(dtype=np.int64)
            vals = labels[idx]
            if len(idx) < window_epochs:
                continue
            for start in range(0, len(idx) - window_epochs + 1, max(window_epochs // 8, 1)):
                chunk_vals = vals[start : start + window_epochs]
                score = shannon_entropy(chunk_vals, n_classes) + 0.08 * len(np.unique(chunk_vals))
                if score > best_score:
                    best_score = score
                    best_indices = idx[start : start + window_epochs]
                    best_label = str(record_id)

    if best_score < 0:
        stride = max(window_epochs // 8, 1)
        for start in range(0, len(labels) - window_epochs + 1, stride):
            chunk_vals = labels[start : start + window_epochs]
            score = shannon_entropy(chunk_vals, n_classes) + 0.08 * len(np.unique(chunk_vals))
            if score > best_score:
                best_score = score
                best_indices = np.arange(start, start + window_epochs)

    return best_indices, best_label


def representative_indices(
    x: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
    candidates_per_class: int = 192,
) -> list[int]:
    chosen = []
    for label in range(n_classes):
        pool = np.flatnonzero(labels == label)
        if len(pool) == 0:
            chosen.append(-1)
            continue
        sample_size = min(candidates_per_class, len(pool))
        candidates = rng.choice(pool, size=sample_size, replace=False)
        rms = []
        for idx in candidates:
            epoch = np.asarray(x[idx], dtype=np.float32)
            val = np.sqrt(np.mean(np.square(epoch), axis=1)).mean()
            rms.append(float(val))
        median_pos = int(np.argsort(np.abs(np.asarray(rms) - np.median(rms)))[0])
        chosen.append(int(candidates[median_pos]))
    return chosen


def select_spectrum_indices(
    labels: np.ndarray,
    n_classes: int,
    samples_per_class: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    selected = []
    for label in range(n_classes):
        pool = np.flatnonzero(labels == label)
        if len(pool) == 0:
            selected.append(np.array([], dtype=np.int64))
            continue
        size = min(samples_per_class, len(pool))
        selected.append(np.sort(rng.choice(pool, size=size, replace=False)))
    return selected


def robust_standardize(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    center = np.nanmedian(signal)
    scale = np.nanpercentile(signal, 95) - np.nanpercentile(signal, 5)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanstd(signal) + 1e-6)
    out = (signal - center) / scale
    return np.clip(out, -3.5, 3.5)


def compute_stage_spectra(
    x: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    samples_per_class: int,
    sample_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    selected = select_spectrum_indices(labels, n_classes, samples_per_class, rng)
    n_channels = x.shape[1]
    n_time = x.shape[2]
    freqs = np.fft.rfftfreq(n_time, d=1.0 / sample_rate)
    mask = (freqs >= 0.5) & (freqs <= 25.0)
    freqs = freqs[mask]
    spectra = np.full((n_channels, n_classes, len(freqs)), np.nan, dtype=np.float32)
    window = np.hanning(n_time).astype(np.float32)

    for label, indices in enumerate(selected):
        if len(indices) == 0:
            continue
        accum = []
        for idx in indices:
            epoch = np.asarray(x[idx], dtype=np.float32)
            epoch = epoch - np.median(epoch, axis=1, keepdims=True)
            scale = np.std(epoch, axis=1, keepdims=True) + 1e-6
            epoch = epoch / scale
            fft = np.fft.rfft(epoch * window[None, :], axis=1)
            power = (np.abs(fft) ** 2)[:, mask]
            accum.append(power)
        mean_power = np.mean(np.stack(accum, axis=0), axis=0)
        spectra[:, label, :] = 10.0 * np.log10(mean_power + 1e-8)
    return freqs, spectra


def plot_class_distribution(ax: plt.Axes, labels: np.ndarray, class_names: list[str]) -> None:
    counts = class_counts(labels, len(class_names))
    total = counts.sum()
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, counts, color=STAGE_COLORS, height=0.68, alpha=0.95)
    ax.set_yticks(y_pos, class_names)
    ax.invert_yaxis()
    ax.set_xlabel("Epoch count")
    ax.set_title("A  Stage distribution", loc="left", pad=8)
    ax.grid(axis="x", color="#d9dde7", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for i, count in enumerate(counts):
        pct = 100.0 * count / max(total, 1)
        ax.text(count + max(counts) * 0.012, i, f"{count:,}  ({pct:.1f}%)", va="center", fontsize=8.5)


def plot_hypnogram(
    ax: plt.Axes,
    labels: np.ndarray,
    indices: np.ndarray,
    class_names: list[str],
    window_label: str,
) -> None:
    stage_map = {label: pos for pos, label in enumerate(range(len(class_names)))}
    y_vals = np.array([stage_map.get(int(v), np.nan) for v in labels[indices]])
    minutes = np.arange(len(indices)) * 0.5
    cmap = ListedColormap(STAGE_COLORS[: len(class_names)])
    ax.imshow(
        y_vals[None, :],
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        extent=[0, minutes[-1] + 0.5, len(class_names) - 0.5, -0.5],
        alpha=0.22,
    )
    ax.step(minutes, y_vals, where="post", color="#111827", linewidth=1.25)
    ax.scatter(minutes[::8], y_vals[::8], c=[STAGE_COLORS[int(v)] for v in labels[indices][::8]], s=12, zorder=3)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_ylim(len(class_names) - 0.5, -0.5)
    ax.set_xlim(0, minutes[-1] + 0.5)
    ax.set_xlabel("Time within selected record (min)")
    ax.set_title(f"B  Continuous hypnogram excerpt ({window_label})", loc="left", pad=8)
    ax.grid(axis="x", color="#d9dde7", linewidth=0.7, alpha=0.65)


def plot_epoch_trace(
    ax: plt.Axes,
    epoch: np.ndarray,
    label_name: str,
    epoch_index: int,
    channel_names: list[str],
    sample_rate: float,
    stage_color: str,
) -> None:
    n_time = epoch.shape[-1]
    time = np.arange(n_time) / sample_rate
    offsets = np.arange(epoch.shape[0])[::-1] * 1.85
    ax.axvspan(time[0], time[-1], color=stage_color, alpha=0.08, linewidth=0)
    for ch, offset in enumerate(offsets):
        signal = robust_standardize(epoch[ch])
        ax.plot(time, signal + offset, color=CHANNEL_COLORS[ch % len(CHANNEL_COLORS)], linewidth=0.72, alpha=0.94)
    ax.set_xlim(0, time[-1])
    ax.set_ylim(offsets[-1] - 0.95, offsets[0] + 0.95)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks(offsets, short_channel_names(channel_names))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#d9dde7", linewidth=0.55, alpha=0.65)
    ax.set_title(f"{label_name}   epoch {epoch_index:,}", color="#111827", pad=7)


def normalize_for_heatmap(data: np.ndarray) -> np.ndarray:
    out = data.copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out
    lo, hi = np.nanpercentile(out[finite], [3, 97])
    out = np.clip(out, lo, hi)
    return out


def plot_spectrum_heatmap(
    ax: plt.Axes,
    freqs: np.ndarray,
    spectrum: np.ndarray,
    class_names: list[str],
    channel_name: str,
    show_ylabel: bool,
    vmin: float,
    vmax: float,
) -> None:
    im = ax.imshow(
        spectrum,
        aspect="auto",
        origin="upper",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        extent=[freqs[0], freqs[-1], len(class_names) - 0.5, -0.5],
    )
    ax.set_title(channel_name)
    ax.set_xlabel("Frequency (Hz)")
    if show_ylabel:
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_ylabel("Sleep stage")
    else:
        ax.set_yticks(range(len(class_names)), [])
    ax.set_xticks([0.5, 4, 8, 12, 16, 20, 25])
    for boundary in [4, 8, 12]:
        ax.axvline(boundary, color="white", linewidth=0.8, alpha=0.45)
    return im


def build_figure(
    x: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    channel_names: list[str],
    data_dir: Path,
    output_dir: Path,
    sample_rate: float,
    seed: int,
    window_epochs: int,
    spectrum_samples: int,
    export_pdf: bool,
    export_svg: bool,
) -> Path:
    rng = np.random.default_rng(seed)
    epoch_info = load_epoch_info(data_dir)
    hyp_indices, hyp_label = choose_hypnogram_window(labels, len(class_names), window_epochs, epoch_info)
    rep_indices = representative_indices(x, labels, len(class_names), rng)
    freqs, spectra = compute_stage_spectra(x, labels, len(class_names), spectrum_samples, sample_rate, rng)

    fig = plt.figure(figsize=(17.6, 12.2), facecolor="#f7f8fb")
    gs = fig.add_gridspec(
        nrows=6,
        ncols=5,
        height_ratios=[0.72, 0.16, 1.0, 1.0, 0.17, 1.05],
        left=0.045,
        right=0.985,
        top=0.91,
        bottom=0.13,
        hspace=0.42,
        wspace=0.36,
    )
    fig.suptitle(
        "Sleep-EDF Processed EEG/EOG Dataset: Multi-Channel Epoch Showcase",
        x=0.045,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.045,
        0.943,
        f"Input tensor {tuple(x.shape)} | five R&K-derived sleep stages | movement/unknown epochs excluded",
        ha="left",
        fontsize=10.5,
        color="#4b5565",
    )

    ax_dist = fig.add_subplot(gs[0, :2])
    plot_class_distribution(ax_dist, labels, class_names)

    ax_hyp = fig.add_subplot(gs[0, 2:])
    plot_hypnogram(ax_hyp, labels, hyp_indices, class_names, hyp_label)

    ax_c_title = fig.add_subplot(gs[1, :])
    ax_c_title.axis("off")
    ax_c_title.text(
        0.0,
        0.5,
        "C  Representative 30-second synchronized epochs",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#111827",
    )

    for i, idx in enumerate(rep_indices):
        ax = fig.add_subplot(gs[2:4, i])
        if idx < 0:
            ax.axis("off")
            ax.set_title(f"{class_names[i]} missing")
            continue
        epoch = np.asarray(x[idx], dtype=np.float32)
        plot_epoch_trace(ax, epoch, class_names[i], idx, channel_names, sample_rate, STAGE_COLORS[i])
        if i == 0:
            ax.set_ylabel("Robust-normalized amplitude")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Time (s)")

    ax_d_title = fig.add_subplot(gs[4, :])
    ax_d_title.axis("off")
    ax_d_title.text(
        0.0,
        0.48,
        "D  Stage-wise spectral fingerprints (mean log power, 0.5-25 Hz)",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#111827",
    )

    heatmap_data = [normalize_for_heatmap(spectra[ch]) for ch in range(x.shape[1])]
    finite_values = np.concatenate([data[np.isfinite(data)].ravel() for data in heatmap_data if np.isfinite(data).any()])
    vmin, vmax = np.percentile(finite_values, [2, 98])

    bottom_gs = gs[5, :].subgridspec(1, x.shape[1], wspace=0.32)
    heatmap_axes = []
    images = []
    for ch in range(x.shape[1]):
        ax = fig.add_subplot(bottom_gs[0, ch])
        heatmap_axes.append(ax)
        im = plot_spectrum_heatmap(
            ax,
            freqs,
            heatmap_data[ch],
            class_names,
            channel_names[ch] if ch < len(channel_names) else f"Channel {ch}",
            show_ylabel=(ch == 0),
            vmin=float(vmin),
            vmax=float(vmax),
        )
        images.append(im)

    cbar_ax = fig.add_axes([0.37, 0.055, 0.26, 0.014])
    cbar = fig.colorbar(images[-1], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Log power (a.u.)", labelpad=2)

    legend_handles = [
        mpl.patches.Patch(facecolor=STAGE_COLORS[i], edgecolor="none", label=class_names[i])
        for i in range(len(class_names))
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.972),
        ncol=len(class_names),
        frameon=False,
        handlelength=1.2,
        columnspacing=1.2,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "sleepedf_dataset_showcase.png"
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    if export_pdf:
        fig.savefig(output_dir / "sleepedf_dataset_showcase.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    if export_svg:
        fig.savefig(output_dir / "sleepedf_dataset_showcase.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize processed Sleep-EDF EEG/EOG epochs.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample_rate", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hypnogram_epochs", type=int, default=240, help="Number of 30 s epochs in the hypnogram excerpt.")
    parser.add_argument("--spectrum_samples", type=int, default=96, help="Epochs sampled per class for spectral fingerprints.")
    parser.add_argument("--no_pdf", action="store_true", help="Skip PDF export.")
    parser.add_argument("--svg", action="store_true", help="Also export SVG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    x, labels = load_arrays(args.data_dir)
    class_names = load_class_names(args.data_dir)
    channel_names = load_channel_names(args.data_dir)
    png_path = build_figure(
        x=x,
        labels=labels,
        class_names=class_names,
        channel_names=channel_names,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        seed=args.seed,
        window_epochs=args.hypnogram_epochs,
        spectrum_samples=args.spectrum_samples,
        export_pdf=not args.no_pdf,
        export_svg=args.svg,
    )
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
