import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
from scipy import interpolate

# PREPARING COLOURS FOR PLOTS
colours=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#aec7e8', '#ffff00', '#00b2ec', '#1f1f77', '#e377c2', '#4e79a7', '#f08080', '#000000']

styles_points=[{'marker':'o', 'c':colours[0]}, {'marker':'*', 'c':colours[1]}, {'marker':'s', 'c':colours[2]}, {'marker':'X', 'c':colours[3]}, 
        {'marker':'D', 'c':colours[4]}, {'marker':'P', 'c':colours[5]}]

styles_lines=[{'linestyle':'-', 'c':colours[0]}, {'linestyle':'-', 'c':colours[1]}, {'linestyle':'-', 'c':colours[2]}, {'linestyle':'-', 'c':colours[3]}, 
        {'linestyle':'-', 'c':colours[4]}, {'linestyle':'-', 'c':colours[5]}]


# READING DATA
data_n=pd.read_csv("results_n.csv")
data_n.rename(columns={'Unnamed: 0': 'size'}, inplace=True)
data_d=pd.read_csv("results_d.csv")
data_d.rename(columns={'Unnamed: 0': 'size'}, inplace=True)
data_k=pd.read_csv("results_k.csv")
data_k.rename(columns={'Unnamed: 0': 'size'}, inplace=True)


# GETTING DATAFRAMES IN THE RIGHT FORMAT
dist_calc_data={}
runtime_data={}
for column_name in data_n.columns:
    if column_name=='size':
        dist_calc_data[column_name]=data_n[column_name]
        runtime_data[column_name]=data_n[column_name]
    else:
        dist_calc_data[column_name]=data_n[column_name].str.split('/').str[0].astype(int)
        runtime_data[column_name]=data_n[column_name].str.split('/').str[1].str.split('ms').str[0].astype(int)/1000

n_distance_calc=pd.DataFrame(dist_calc_data)
n_runtime=pd.DataFrame(runtime_data)

dist_calc_data={}
runtime_data={}
for column_name in data_d.columns:
    if column_name=='size':
        dist_calc_data[column_name]=data_d[column_name]
        runtime_data[column_name]=data_d[column_name]
    else:
        dist_calc_data[column_name]=data_d[column_name].str.split('/').str[0].astype(int)
        runtime_data[column_name]=data_d[column_name].str.split('/').str[1].str.split('ms').str[0].astype(int)/1000

d_distance_calc=pd.DataFrame(dist_calc_data)
d_runtime=pd.DataFrame(runtime_data)

dist_calc_data={}
runtime_data={}
for column_name in data_k.columns:
    if column_name=='size':
        dist_calc_data[column_name]=data_k[column_name]
        runtime_data[column_name]=data_k[column_name]
    else:
        dist_calc_data[column_name]=data_k[column_name].str.split('/').str[0].astype(int)
        runtime_data[column_name]=data_k[column_name].str.split('/').str[1].str.split('ms').str[0].astype(int)/1000

k_distance_calc=pd.DataFrame(dist_calc_data)
k_runtime=pd.DataFrame(runtime_data)


## PLOTS
plt.figure(figsize=(24, 16))
plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.93, wspace=0.2, hspace=0.15)
G = gridspec.GridSpec(2, 4)
ax1 = plt.subplot(G[0, 0:2])
ax2 = plt.subplot(G[0, 2:4])
ax3 = plt.subplot(G[1, 1:3])

n_new=np.linspace(100, 1000000, 100)
d_new=np.linspace(2, 1000, 100)
k_new=np.linspace(5, 1000, 100)

# FIRST PLOT
column_index=1
for column_name in n_runtime.columns:
    if column_name=='size':
        continue
    else:
        if column_name=='Lloyd':
            column_index+=1
            continue
        bspline=interpolate.make_interp_spline(n_runtime['size'], n_runtime[column_name], k=1)
        y_interpolated=bspline(n_new)
        ax1.scatter(n_runtime['size'], n_runtime[column_name], s=100, label=n_runtime.columns[column_index], **styles_points[column_index-1])
        ax1.plot(n_new, y_interpolated, **styles_lines[column_index-1])
        column_index+=1

ax1.legend(fontsize="14")
ax1.set_title("Sample size", size=25)
ax1.set_ylabel("Time to fit (s)", fontsize=14)
ax1.set_xlabel("Data samples", fontsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=14)

# SECOND PLOT
column_index=1
for column_name in d_runtime.columns:
    if column_name=='size':
        continue
    else:
        if column_name=='Lloyd':
            column_index+=1
            continue
        bspline=interpolate.make_interp_spline(d_runtime['size'], d_runtime[column_name], k=1)
        y_interpolated=bspline(d_new)
        ax2.scatter(d_runtime['size'], d_runtime[column_name], s=100, label=d_runtime.columns[column_index], **styles_points[column_index-1])
        ax2.plot(d_new, y_interpolated, **styles_lines[column_index-1])
        column_index+=1

ax2.legend(fontsize="14")
ax2.set_title("Dimensionality", size=25)
ax2.set_ylabel("Time to fit (s)", fontsize=14)
ax2.set_xlabel("Number of dimensions", fontsize=14)
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=14)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=14)

# THIRD PLOT
column_index=1
for column_name in k_runtime.columns:
    if column_name=='size':
        continue
    else:
        if column_name=='Lloyd':
            column_index+=1
            continue
        bspline=interpolate.make_interp_spline(k_runtime['size'], k_runtime[column_name], k=1)
        y_interpolated=bspline(k_new)
        ax3.scatter(k_runtime['size'], k_runtime[column_name], s=100, label=k_runtime.columns[column_index], **styles_points[column_index-1])
        ax3.plot(k_new, y_interpolated, **styles_lines[column_index-1])
        column_index+=1

ax3.legend(fontsize="14")
ax3.set_title("Centroids", size=25)
ax3.set_ylabel("Time to fit (s)", fontsize=14)
ax3.set_xlabel("Number of centroids", fontsize=14)
ax3.set_xticklabels(ax3.get_xticklabels(), fontsize=14)
ax3.set_yticklabels(ax3.get_yticklabels(), fontsize=14)

plt.savefig('plot_runtime.pdf')