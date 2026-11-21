"""Draw selected operating-history, count-calibration and GW-interval comparisons."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paths import RES


def main():
    trajectory = pd.read_csv(RES / "pre_retirement_trajectories.csv")
    scores = pd.read_csv(RES / "backtest.csv").query("population == 'Any operation'")
    intervals = pd.read_csv(RES / "fleet_interval_calibration_summary.csv")
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for fuel, colour in [("coal", "#c37e75"), ("gas", "#7299bc")]:
        data = trajectory[(trajectory.fuel == fuel) & (trajectory.metric == "cf")].sort_values("rel")
        axes[0].plot(data.rel, data.median_excess, color=colour, label=fuel.title())
        axes[0].fill_between(data.rel, data.lo, data.hi, color=colour, alpha=0.22)
    axes[0].axhline(0, color="0.45", lw=0.8, ls=":")
    axes[0].set(xlabel="Years before final operation", ylabel="Capacity factor difference")
    for model, label, marker in [("Hazard (pooled)", "Pooled hazard", "o"),
                                 ("Classifier LightGBM", "Fixed-horizon classifier", "s")]:
        sums = scores[scores.model == model].groupby("H")[["exp", "obs"]].sum()
        axes[1].plot(sums.index, sums.exp / sums.obs, marker=marker, label=label)
    axes[1].axhline(1, color="0.45", lw=0.8, ls=":")
    axes[1].set(xlabel="Forecast horizon (years)", ylabel="Predicted / observed cessations", xticks=[1, 3, 5])
    for rule, label, marker in [("Independent units", "Independent", "s"),
                                ("Sequential wave-calibrated", "Sequential", "o")]:
        rows = intervals[intervals.rule == rule]
        axes[2].plot(rows.H, 100 * rows.coverage_90, marker=marker, label=label)
    axes[2].axhline(90, color="0.45", lw=0.8, ls=":")
    axes[2].set(xlabel="Forecast horizon (years)", ylabel="Observed capacity inside interval (%)",
                xticks=[1, 3, 5], ylim=(0, 105))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.savefig(RES / "core_comparisons.pdf", metadata={"Author": "CHONG LIU"})
    plt.close(figure)


if __name__ == "__main__":
    main()
