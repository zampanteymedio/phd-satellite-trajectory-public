import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def main(argv):
    parser = argparse.ArgumentParser(description='Plot the results of the training.')
    parser.add_argument('--dir', '-d', metavar='D', type=str,
                        help='directory containing evaluations')
    args = parser.parse_args(argv)
    process(**vars(args))


def process(dir):
    files = map(lambda x: os.path.join(dir, x, "evaluations.npz"), os.listdir(dir))
    timesteps = []
    results = []
    for file in files:
        data = np.load(file)
        timesteps.append(data["timesteps"].transpose())
        results.append(data["results"].transpose())

    timesteps = np.stack(timesteps)
    results = np.stack(results)
    p10 = np.percentile(results, 10, axis=0)
    p30 = np.percentile(results, 30, axis=0)
    p50 = np.percentile(results, 50, axis=0)
    p70 = np.percentile(results, 70, axis=0)
    p90 = np.percentile(results, 90, axis=0)

    fig, axs = plt.subplots(1, 1, figsize=(4.8, 3.0))
    axs.set_xlim(1000, 500000)
    axs.set_ylim(-0.2, 1.75)
    axs.grid(True)
    axs.set_xlabel("number of steps")
    axs.set_ylabel("evaluation reward")
    axs.plot(timesteps[0], p50[0], "k")
    axs.fill_between(timesteps[0], p30[0], p70[0], alpha=0.2, facecolor="black")
    axs.fill_between(timesteps[0], p10[0], p90[0], alpha=0.1, facecolor="black")
    axs.set_xscale('log')
    axs.legend(["Median", "Pct40", "Pct80"])
    plt.tight_layout()
    fig.savefig("training.pdf", format="pdf")
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1:])
