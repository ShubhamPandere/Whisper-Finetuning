import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Whisper QLoRA training logs.")
    parser.add_argument("--log-files", nargs="+", required=True, help="Paths to training_logs.csv files.")
    parser.add_argument("--labels", nargs="+", default=None, help="Plot labels for the log files.")
    parser.add_argument("--output-dir", default=None, help="Directory for saved plots.")
    return parser.parse_args()


def pick_label(path, index, labels):
    if labels and index < len(labels):
        return labels[index]
    return Path(path).parent.name


def plot_metric(log_files, labels, metric_name, title, ylabel, output_path):
    plt.figure(figsize=(10, 6))
    for index, log_file in enumerate(log_files):
        df = pd.read_csv(log_file)
        if "step" not in df.columns or metric_name not in df.columns:
            continue
        label = pick_label(log_file, index, labels)
        df = df.dropna(subset=["step", metric_name])
        if df.empty:
            continue
        plt.plot(df["step"], df[metric_name], marker="o", linewidth=2, label=label)

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        args.log_files,
        args.labels,
        "loss",
        "Training Loss",
        "Loss",
        output_dir / "training_loss.png",
    )
    plot_metric(
        args.log_files,
        args.labels,
        "eval_loss",
        "Validation Loss",
        "Eval Loss",
        output_dir / "validation_loss.png",
    )
    plot_metric(
        args.log_files,
        args.labels,
        "eval_wer",
        "Validation WER",
        "WER (%)",
        output_dir / "validation_wer.png",
    )

    print(f"Saved plots in {output_dir}")


if __name__ == "__main__":
    main()
