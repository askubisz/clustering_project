import numpy as np
import pandas as pd
from itertools import cycle, islice
import matplotlib.pyplot as plt

colours=['#000000', '#2ca02c', '#9467bd', '#8c564b', '#00b2ec', '#1f1f77', '#e377c2', '#4e79a7', '#f08080']

# styles_lines=[{'c':colours[0]}, {'c':colours[1]}, {'c':colours[2]}, {'c':colours[3]}, 
#         {'c':colours[4]}, {'c':colours[5]}, {'c':colours[6]}, {'c':colours[7]}, 
#         {'c':colours[8]}, {'c':colours[9]}, {'c':colours[10]}, {'c':colours[11]}, 
#         {'c':colours[12]}, {'c':colours[13]}]

fcps_df=pd.read_csv('fcps_extrinsic.csv', index_col=0)
sipu_df=pd.read_csv('sipu_extrinsic.csv', index_col=0)
graves_df=pd.read_csv('graves_extrinsic.csv', index_col=0)
uci_df=pd.read_csv('uci_extrinsic.csv', index_col=0)

# GETTING DATAFRAMES IN THE RIGHT FORMAT
ari_data={}
for data_frame in [fcps_df, sipu_df, graves_df, uci_df]:
    for dataset_name in data_frame.index.tolist():
        if dataset_name=='engytime':
            for algo_name in data_frame.columns:
                if algo_name=='Unnamed: 0':
                    continue
                ari_data[algo_name]=[float(data_frame.loc[dataset_name][algo_name].split('/')[0])]
        if dataset_name=='flame':
            for algo_name in data_frame.columns:
                if algo_name=='Unnamed: 0':
                    continue
                ari_data[algo_name].append(float(data_frame.loc[dataset_name][algo_name].split('/')[0]))
        if dataset_name=='dense':
            for algo_name in data_frame.columns:
                if algo_name=='Unnamed: 0':
                    continue
                ari_data[algo_name].append(float(data_frame.loc[dataset_name][algo_name].split('/')[0]))
        if dataset_name=='wine':
            for algo_name in data_frame.columns:
                if algo_name=='Unnamed: 0':
                    continue
                ari_data[algo_name].append(float(data_frame.loc[dataset_name][algo_name].split('/')[0]))


ari_df=pd.DataFrame(ari_data)
ari_df=ari_df[['KM', 'DBSCAN', 'HDBSCAN', 'MS', 'Ward', 'Spectral', 'CLASSIX', 'DenMune', 'FINCH']]

### PLOT
plt.figure(figsize=(24, 16))
plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.93, wspace=0.1, hspace=0.25)

plt.subplot(2,2,1)
plt.bar(ari_df.columns.tolist(), ari_df.iloc[0,:].values, color=colours, width = 0.4)
plt.ylabel("AR index", fontsize=15)
plt.xlabel("Algorithm", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('engytime (high overlap + spherical)', fontsize=25)

plt.subplot(2,2,2)
plt.bar(ari_df.columns.tolist(), ari_df.iloc[1,:].values, color=colours, width = 0.4)
plt.ylabel("AR index", fontsize=15)
plt.xlabel("Algorithm", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('flame (adjacent + non standard shape)', fontsize=25)


plt.subplot(2,2,3)
plt.bar(ari_df.columns.tolist(), ari_df.iloc[2,:].values, color=colours, width = 0.4)
plt.ylabel("AR index", fontsize=15)
plt.xlabel("Algorithm", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('dense (varying density + spherical)', fontsize=25)


plt.subplot(2,2,4)
plt.bar(ari_df.columns.tolist(), ari_df.iloc[3,:].values, color=colours, width = 0.4)
plt.ylabel("AR index", fontsize=15)
plt.xlabel("Algorithm", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('wine (real world dataset)', fontsize=25)


plt.savefig('barplot.pdf')