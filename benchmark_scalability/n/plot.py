import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import interpolate


colours=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#aec7e8', '#ffff00', '#00b2ec', '#1f1f77', '#e377c2', '#4e79a7', '#f08080', '#000000']

data=pd.read_csv("times.csv")
data.rename(columns={'Unnamed: 0': 'size'}, inplace=True)

plt.figure(figsize=(24, 16))
plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.93, wspace=0.15, hspace=0.15)
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

# PLOT 1
x1_new=np.linspace(100, 8000, 100)
data_small=data[0:5]

indexes_ordered_small=[6,12,4,10,11,9,8,13,2,5,7,3,0,1]
indexes_ordered_medium=[4,10,12,11,5,7,3,0,1]
indexes_ordered_large=[5,7,3,0,1]


for column_index in indexes_ordered_small:
    bspline=interpolate.make_interp_spline(data_small['size'], data_small.iloc[:, column_index+1], k=1)
    y_interpolated=bspline(x1_new)
    ax1.scatter(data_small['size'], data_small.iloc[:, column_index+1], s=100, label=data_small.columns[column_index+1], **styles_points[column_index-1])
    ax1.plot(x1_new, y_interpolated, **styles_lines[column_index-1])

data_medium=data[0:9]

for column_index in indexes_ordered_medium:
    if data.columns[column_index+1]=='PAM' or data.columns[column_index+1]=='MS' or data.columns[column_index+1]=='Comp' or data.columns[column_index+1]=='Ward' or data.columns[column_index+1]=='FINCH':
        continue
    elif data.columns[column_index+1]=='OPTICS' or data.columns[column_index+1]=='Spectral' or data.columns[column_index+1]=='DenMune':
        x2_new=np.linspace(100, 32000, 100)
        bspline=interpolate.make_interp_spline(data_medium['size'][:7], data_medium.iloc[:7, column_index+1], k=1)
        y_interpolated=bspline(x2_new)
        ax2.scatter(data_medium['size'][:7], data_medium.iloc[:7, column_index+1], s=100, label=data_medium.columns[column_index+1], **styles_points[column_index-1])
        ax2.plot(x2_new, y_interpolated, **styles_lines[column_index-1])
    elif data.columns[column_index+1]=='CLASSIX':
        x2_new=np.linspace(100, 50000, 100)
        bspline=interpolate.make_interp_spline(data_medium['size'][:8], data_medium.iloc[:8, column_index+1], k=1)
        y_interpolated=bspline(x2_new)
        ax2.scatter(data_medium['size'][:8], data_medium.iloc[:8, column_index+1], s=100, label=data_medium.columns[column_index+1], **styles_points[column_index-1])
        ax2.plot(x2_new, y_interpolated, **styles_lines[column_index-1])
    else:
        x2_new=np.linspace(100, 75000, 100)
        bspline=interpolate.make_interp_spline(data_medium['size'], data_medium.iloc[:, column_index+1], k=1)
        y_interpolated=bspline(x2_new)
        ax2.scatter(data_medium['size'], data_medium.iloc[:, column_index+1], s=100, label=data_medium.columns[column_index+1], **styles_points[column_index-1])
        ax2.plot(x2_new, y_interpolated, **styles_lines[column_index-1])
    

data_large=data

for column_index in indexes_ordered_large:
    try:
        if data.columns[column_index+1]=='HDBSCAN' or data.columns[column_index+1]=='Single':
            
            x3_new=np.linspace(100, 100000, 100)
            bspline=interpolate.make_interp_spline(data_large['size'][:10], data_large.iloc[:10, column_index+1], k=1)
            y_interpolated=bspline(x3_new)
            ax3.scatter(data_large['size'][:10], data_large.iloc[:10, column_index+1], s=100, label=data_large.columns[column_index+1], **styles_points[column_index-1])
            ax3.plot(x3_new, y_interpolated, **styles_lines[column_index-1])
        else:
            x3_new=np.linspace(100, 1000000, 100)
            bspline=interpolate.make_interp_spline(data_large['size'], data_large.iloc[:, column_index+1], k=1)
            y_interpolated=bspline(x3_new)
            ax3.scatter(data_large['size'], data_large.iloc[:, column_index+1], s=100, label=data_large.columns[column_index+1], **styles_points[column_index-1])
            ax3.plot(x3_new, y_interpolated, **styles_lines[column_index-1])
    except:
        continue

ax1.legend(fontsize="14")
ax1.set_title("a)", size=25)
ax1.set_ylabel("Time to fit (s)", fontsize=14)
ax1.set_xlabel("Data samples", fontsize=14)
ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=14)
ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=14)

# PLOT 2

ax2.set_title("b)", size=25)
ax2.legend(fontsize="14")
ax2.set_ylabel("Time to fit (s)", fontsize=14)
ax2.set_xlabel("Data samples", fontsize=14)
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=14)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=14)


# PLOT 3

ax3.legend(fontsize="14")
ax3.set_title("c)", size=25)
ax3.set_ylabel("Time to fit (s)", fontsize=14)
ax3.set_xlabel("Data samples", fontsize=14)
ax3.tick_params(axis='x', labelsize=14)
ax3.set_yticklabels(ax3.get_yticklabels(), fontsize=14)

plt.suptitle('Runtime comparison for different number of samples with set d=10 and k=10', size=25)
plt.savefig('scalability_plot_n.pdf')

