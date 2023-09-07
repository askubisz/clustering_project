import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import interpolate


colours=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#aec7e8', '#ffff00', '#00b2ec', '#1f1f77', '#e377c2', '#4e79a7', '#f08080', '#000000']

data=pd.read_csv("times.csv")
data.rename(columns={'Unnamed: 0': 'size'}, inplace=True)

plt.figure(figsize=(24, 16))
plt.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.93, wspace=0.15, hspace=0.15)
G = gridspec.GridSpec(2, 4)
ax1 = plt.subplot(G[0, 0:2])
ax2 = plt.subplot(G[0, 2:4])
ax3 = plt.subplot(G[1, 1:3])

styles_points=[{'marker':'o', 'c':colours[0]}, {'marker':'o', 'c':colours[1]}, {'marker':'o', 'c':colours[2]}, {'marker':'*', 'c':colours[3]}, 
        {'marker':'*', 'c':colours[4]}, {'marker':'*', 'c':colours[5]}, {'marker':'*', 'c':colours[6]}, {'marker':'s', 'c':colours[7]}, 
        {'marker':'s', 'c':colours[8]}, {'marker':'s', 'c':colours[9]}, {'marker':'s', 'c':colours[10]}, {'marker':'X', 'c':colours[11]}, 
        {'marker':'X', 'c':colours[12]}, {'marker':'o', 'c':colours[13]}]

styles_lines=[{'linestyle':'-', 'c':colours[0]}, {'linestyle':'-', 'c':colours[1]}, {'linestyle':'-', 'c':colours[2]}, {'linestyle':'-', 'c':colours[3]}, 
        {'linestyle':'-', 'c':colours[4]}, {'linestyle':'-', 'c':colours[5]}, {'linestyle':'-', 'c':colours[6]}, {'linestyle':'-', 'c':colours[7]}, 
        {'linestyle':'-', 'c':colours[8]}, {'linestyle':'-', 'c':colours[9]}, {'linestyle':'-', 'c':colours[10]}, {'linestyle':'-', 'c':colours[11]}, 
        {'linestyle':'-', 'c':colours[12]}, {'linestyle':'-', 'c':colours[13]}]

indexes_ordered_small=[4,11,12,6,10,9,5,8,13,2,7,1,0,3]
indexes_ordered_medium=[5,7,9,8,10,13,2,3,1,0]
indexes_ordered_large=[13,2,3,1,0]

# PLOT 1
x1_new=np.linspace(2, 25, 100)
data_small=data[0:4]

for column_index in indexes_ordered_small:
    bspline=interpolate.make_interp_spline(data_small['size'], data_small.iloc[:, column_index+1], k=1)
    y_interpolated=bspline(x1_new)
    ax1.scatter(data_small['size'], data_small.iloc[:, column_index+1], s=100, label=data_small.columns[column_index+1], **styles_points[column_index-1])
    ax1.plot(x1_new, y_interpolated, **styles_lines[column_index-1])

for column_index in indexes_ordered_medium:
    try:
        if data.columns[column_index+1]=='FINCH' or data.columns[column_index+1]=='PAM' or data.columns[column_index+1]=='DBSCAN' or data.columns[column_index+1]=='KM' or data.columns[column_index+1]=='MBKM':
            continue
        else:
            x2_new=np.linspace(2, 1000, 100)
            bspline=interpolate.make_interp_spline(data['size'], data.iloc[:, column_index+1], k=1)
            y_interpolated=bspline(x2_new)
            ax2.scatter(data['size'], data.iloc[:, column_index+1], s=100, label=data.columns[column_index+1], **styles_points[column_index-1])
            ax2.plot(x2_new, y_interpolated, **styles_lines[column_index-1])
    except:
        continue

    

for column_index in indexes_ordered_large:
    try:
        if data.columns[column_index+1]=='HDBSCAN' or data.columns[column_index+1]=='Single' or data.columns[column_index+1]=='Ward' or data.columns[column_index+1]=='Comp' or data.columns[column_index+1]=='Spectral':
            continue
        else:
            x3_new=np.linspace(2, 1000, 100)
            bspline=interpolate.make_interp_spline(data['size'], data.iloc[:, column_index+1], k=1)
            y_interpolated=bspline(x3_new)
            ax3.scatter(data['size'], data.iloc[:, column_index+1], s=100, label=data.columns[column_index+1], **styles_points[column_index-1])
            ax3.plot(x3_new, y_interpolated, **styles_lines[column_index-1])
    except:
        continue


ax1.legend(fontsize="14")
ax1.set_title("a)", size=25)
ax1.set_ylabel("Time to fit (s)", fontsize=14)
ax1.set_xlabel("Number of dimensions", fontsize=14)
ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=14)
ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=14)

# PLOT 2

ax2.set_title("b)", size=25)
ax2.legend(fontsize="14")
ax2.set_ylabel("Time to fit (s)", fontsize=14)
ax2.set_xlabel("Number of dimensions", fontsize=14)
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=14)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=14)


# PLOT 3

ax3.legend(fontsize="14")
ax3.set_title("c)", size=25)
ax3.set_ylabel("Time to fit (s)", fontsize=14)
ax3.set_xlabel("Number of dimensions", fontsize=14)
ax3.set_xticklabels(ax3.get_xticklabels(), fontsize=14)
ax3.set_yticklabels(ax3.get_yticklabels(), fontsize=14)


plt.suptitle('Runtime comparison for different number of dimensions with set n=5000 and k=10', size=25)
plt.savefig('scalability_plot_d.pdf')
