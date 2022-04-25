import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BivariateSpline

x, e = np.loadtxt("tmp.dat", unpack=True)

idx = np.argsort(x)
x = x[idx]
e = e[idx]

pes = np.polynomial.polynomial.Polynomial.fit(x, e, 25)
xnew = np.linspace(x.min(), x.max(), 1000)
ynew = pes(xnew)

plt.xlim((0.09, 0.875))
plt.xlabel("x", fontweight="bold", fontsize=20)
plt.ylabel("E(x)", fontweight="bold", fontsize=20)
plt.tick_params(
    which="both", bottom=False, left=False, labelbottom=False, labelleft=False
)
plt.plot(xnew, ynew, linewidth=3, color="#5E81AC")
plt.savefig("intro-pes.png", dpi=600)
plt.show()
