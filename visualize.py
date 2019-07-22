import pandas
import matplotlib.pyplot as plt
import tqdm
import os
from DLA import DLA2D
import numpy as np


def plot_all_dimensions(directory=".", x=(5, 80)):

    import glob

    fractals = []

    for filename in glob.glob(os.path.join(directory, "*.json")):
        d = DLA2D.load(filename)
        if d.N_particles >= 1e5:
            fractals.append(d)

    global_min_distance = np.inf
    global_max_distance = -np.inf
    for fractal in fractals:
        distances = fractal.get_off_center_distances()
        min_distance = distances.min()
        if min_distance < global_min_distance:
            global_min_distance = min_distance
        max_distance = distances.max()
        if global_max_distance < max_distance:
            global_max_distance = max_distance

    dataframes = []
    for fractal in tqdm.tqdm(fractals):
        Rs, Ns = fractal.plot_mass_distribution(
            min_distance=global_min_distance,
            max_distance=global_max_distance,
            plot=False,
        )
        dataframes.append(pandas.DataFrame({"R": Rs, "N": Ns}))
    meandf = sum(dataframes) / len(dataframes)
    off_center_dataframes = [(df - meandf) ** 2 for df in dataframes]
    stddf = np.sqrt(sum(off_center_dataframes)) / len(dataframes)
    # meandf.plot('R', 'N', logy=True, logx=True)
    # plt.show()

    xmin, xmax = x
    indices = (xmin < meandf["R"]) & (meandf["R"] < xmax)
    middle = meandf[indices]
    middlestd = stddf[indices]
    (a, b), cov = np.polyfit(np.log10(middle["R"]), np.log10(middle["N"]), 1, cov=True)
    stda = cov[0, 0] ** 0.5
    meandf["Nfit"] = 10 ** np.polyval((a, b), np.log10(meandf["R"]))
    fig, ax = plt.subplots()
    meandf.plot("R", "N", logx=True, logy=True, ax=ax, label="<N>")
    meandf.plot("R", "Nfit", logx=True, logy=True, ax=ax)
    ax.set_title(fr"a = {a:.5f} $\pm$ {stda:.4f}")
    ax.axvline(xmin)
    ax.axvline(xmax)
    for seed, df in enumerate(dataframes):
        df.plot("R", "N", logx=True, logy=True, ax=ax, alpha=0.1, label=f"Seed: {seed}")
    plt.savefig("full_plot.png")
    plt.show()

    return dataframes


def plot_all(directory="."):
    import glob

    for filename in tqdm.tqdm(glob.glob(os.path.join(directory, "*.json"))):
        d = DLA2D.load(filename)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        d.plot_particles(filename.replace(".json", "_particles.png"))
        d.plot_mass_distribution(filename.replace(".json", "_distribution.png"))
        # fig.savefig(dpi=600, fname=filename.replace("json", "png"))


if __name__ == "__main__":
    plot_all()
    plot_all_dimensions(x=(2, 180))
