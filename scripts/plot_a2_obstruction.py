"""Plot exact witness formula, not numerical estimates of sharp CP thresholds."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path("images")
out.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
for q in [1, 2, 4, 9]:
    times = np.linspace(0, 1.7, 501)
    r = np.exp(-times)
    # P_q times the eigenvalue: this rescaling is explicitly shown on the axis.
    witness = (1-r)*(1+q*r)*(1-np.sqrt(q)*r)
    axes[0].plot(times, witness, label=f"q={q}")
axes[0].axhline(0, color="black", linewidth=.7)
axes[0].set(xlabel="time t", ylabel=r"$P_q\,\lambda_-(\Phi_t(p_+))$",
            title="Exact A₂ positivity witness")
axes[0].legend(frameon=False)
qvals = np.geomspace(1, 30, 400)
axes[1].plot(qvals, np.log(qvals)/2, color="#9b2939")
axes[1].fill_between(qvals, 0, np.log(qvals)/2, alpha=.2, color="#9b2939")
axes[1].set(xlabel="Hecke parameter q", ylabel="time t",
            title="Proved failure region (q > 1)")
axes[1].text(10, .4, "not positive", color="#9b2939")
axes[1].text(6, 1.55, "outside region: this witness\ndoes not decide positivity", fontsize=8)
for suffix in ["pdf", "png"]:
    fig.savefig(out/f"a2_obstruction.{suffix}", dpi=180)
print("Saved images/a2_obstruction.pdf and .png")
