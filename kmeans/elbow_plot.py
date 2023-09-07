import numpy as np
from sklearn.cluster import KMeans
import clustbench
import matplotlib.pyplot as plt


# Load data
data=clustbench.load_dataset('sipu', 's1')
data=data.data

inertia=[]
possible_k=list(range(1, 31))

for i in possible_k:
    model=KMeans(i, init='k-means++').fit(data)
    inertia.append(model.inertia_)

plt.figure(figsize=(18, 10))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.2, hspace=0.07)

plt.subplot(1, 2, 1)
plt.plot(possible_k, inertia)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('WCSS', fontsize=15)
plt.xlabel('Number of clusters', fontsize=15)
plt.title("S1 dataset (small overlap)", size=18)


# Load data
data=clustbench.load_dataset('sipu', 's4')
data=data.data

inertia=[]

for i in possible_k:
    model=KMeans(i, init='k-means++').fit(data)
    inertia.append(model.inertia_)



plt.subplot(1, 2, 2)

plt.plot(possible_k, inertia)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('WCSS', fontsize=15)
plt.xlabel('Number of clusters', fontsize=15)
plt.title("S4 dataset (high overlap)", size=18)


plt.savefig('elbow_plot.pdf')