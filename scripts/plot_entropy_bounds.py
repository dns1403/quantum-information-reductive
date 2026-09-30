"""Analytical bounds and explicit noise-rate normalization, not numerical optima."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path("images")
out.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
primes = np.array([2, 3, 5, 7, 11, 17, 31, 61, 127, 251, 509, 1021])
lower = 1/(.5*np.log(primes)+np.log(22))
axes[0].semilogx(primes, lower, "o-", label="proved lower bound: every I ⊂ K")
axes[0].axhline(2, color="#9b2939", label="sharp for spherical and Iwahori")
axes[0].fill_between(primes, lower, 2, alpha=.12)
axes[0].set(xlabel="prime p", ylabel="complete entropy exponent α",
            title="Tree Hecke corners, unit edge length", ylim=(0, 2.2))
axes[0].legend(frameon=False, fontsize=8)
n = np.arange(2, 101)
axes[1].plot(n, 2*np.ones_like(n), label="one rate per edge")
axes[1].plot(n, 4/(n-1), label="total phase-flip event rate = 1")
axes[1].set(xlabel="vertices N in finite Schur model", ylabel="sharp entropy exponent",
            title="Resource normalization changes the rate", ylim=(0, 4.2))
axes[1].legend(frameon=False, fontsize=8)
for suffix in ["pdf", "png"]:
    fig.savefig(out/f"entropy_bounds.{suffix}", dpi=180)
print("Saved images/entropy_bounds.pdf and .png")
