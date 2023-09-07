import numpy as np
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(14)

C1 = np.column_stack((np.random.uniform(-5,5,1000), np.random.uniform(-5,5,1000))) # Noise
C2 = [0, 0.5] + np.random.randn(500, 2)
C3 = [2.5, -3] + 0.6 * np.random.randn(500, 2)
C4 = [-3, 3] + 0.7 * np.random.randn(300, 2)
C5 = [-3, -3] + 0.1 * np.random.randn(50, 2)
X = np.vstack((C1, C2, C3, C4, C5))

# Fit model
X_model=OPTICS(min_samples=5, xi=0.1, metric='euclidean').fit(X)
X_model2=OPTICS(min_samples=15, xi=0.1, metric='euclidean').fit(X)
X_model3=OPTICS(min_samples=50, xi=0.1, metric='euclidean').fit(X)

space_X = np.arange(len(X))
reachability_X = X_model.reachability_[X_model.ordering_]
reachability2_X = X_model2.reachability_[X_model2.ordering_]
reachability3_X = X_model3.reachability_[X_model3.ordering_]


plt.figure(figsize=(12, 8))
plt.rcParams.update({
    "text.usetex": True
})
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# plt.subplot(1,3,1)
ax1.scatter(X[:, 0], X[:, 1], s=20)
ax1.set_title("Synthetic data", size=12)
ax1.set_xticks(())
ax1.set_yticks(())

# plt.subplot(1,3,2)
ax2.plot(space_X, reachability_X, 'black')
ax2.set_title(r'Reachability plot for $MinPts=5$', size=12)
ax2.set_ylabel("Reachability distance")
ax2.set_xlabel("Points ordered by OPTICS")

# plt.subplot(1,3,3)
ax3.plot(space_X, reachability2_X, 'black')
ax3.set_title(r'Reachability plot for $MinPts=15$', size=12)
ax3.set_xlabel("Points ordered by OPTICS")

ax4.plot(space_X, reachability3_X, 'black')
ax4.set_title(r'Reachability plot for $MinPts=50$', size=12)
ax4.set_xlabel("Points ordered by OPTICS")


plt.tight_layout()
plt.savefig("reachability.pdf")