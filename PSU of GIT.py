################################################################################################################################################
### Project  : PSU
### Script   : PSU on GIT.py
### Contents : PSU : Particle Stacking Undersampling for Highly Imblanced Data
################################################################################################################################################

################################################################################################################################################
### Setting up environment
################################################################################################################################################
# Load library
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import StratifiedKFold

# Undersampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import TomekLinks

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce # 2.1.0 version!!
from scipy.spatial import distance
import scipy.stats as ss
import sys
mod = sys.modules[__name__]
import pickle

# Load Datasets
with open("Data.pickle","rb") as fr:
    Dataset = pickle.load(fr)

################################################################################################################################################
### PSU
################################################################################################################################################
def PSU(aa,bb):
    if type(bb) != list:
        bb = bb.ravel()
        if 0 in bb:
            bb[np.where(bb == 0)] = -1
    
    maj_data = aa[bb== -1]
    min_data = aa[bb== 1]
    
    n_maj = len(maj_data)
    n_min = len(min_data)
    
    temp_dist0 = distance.cdist([np.mean(maj_data, axis = 0)], maj_data)
    temp_sort = np.argsort(temp_dist0)
    
    # devide major class data into number of minor class data
    k, m = divmod(len(range(n_maj)), n_min)
    temp_interval = list(range(n_maj)[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_min))
    
    temp_cent = [maj_data[temp_sort[0][temp_interval[0]]][-1]]
    
    for i in range(n_min-1):
        temp_cent.append(maj_data[temp_sort[0][temp_interval[i+1]]][np.argmax(np.sum(distance.cdist(temp_cent, maj_data[temp_sort[0][temp_interval[i+1]]]), axis = 0))])
        
    return [np.vstack((temp_cent, min_data)), np.hstack((-np.ones(n_min), np.ones(n_min)))]

################################################################################################################################################
### Analysis
################################################################################################################################################
## Figure 1.
all_time = Dataset[1]
plt.xlim(9000, 101000)
plt.ylim(np.min(all_time)/2, np.max(all_time)*2)
plt.yscale("log")

labels = ["RUS", "NM-1", "NM-2", "CC","ENN", "CNN", "TomekLinks", "PSU"]
    
for i in range(8):
    plt.scatter(list(range(10000,110000,10000)), all_time[i])
    plt.plot(list(range(10000,110000,10000)), all_time[i], label = labels[i])
    
handles, labels = plt.gca().get_legend_handles_labels()
order = [5,3,2,7,4,6,1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.xlabel("Data Size", fontsize = 15)
plt.ylabel("Resampling Time (s)", fontsize = 15)
plt.show()

## Figure 2.
for i0 in [50]:
    for i1 in range(9):
        seed = 0
        rng = np.random.RandomState(seed)
        
        n_minor = i0
        data_minor = np.r_[2.5 * rng.randn(n_minor, 2) + [5, -10],
                           2.5 * rng.randn(n_minor, 2) + [-5, 10]]
        
        n_major = 1000
        data_major = np.r_[3 * rng.randn(n_major, 2)]
        n_major_outlier = 4
        far = 10
        
        data_major = np.vstack((data_major, 5 * rng.randn(n_major_outlier, 2) + [far,far]))
        data_major = np.vstack((data_major, 5 * rng.randn(n_major_outlier, 2) + [-far,far]))
        data_major = np.vstack((data_major, 5 * rng.randn(n_major_outlier, 2) + [far,-far]))
        data_major = np.vstack((data_major, 5 * rng.randn(n_major_outlier, 2) + [-far,-far]))
        data_major = np.vstack((data_major, 5 * rng.randn(5, 2) + [0,-far*2]))
        
        y = np.zeros(len(data_major))
        X = np.vstack((data_major, data_minor))
        y = np.hstack((y,np.ones(len(data_minor))))
        
        if i1 not in [0,8] :
            if i1 == 1:
                res = RandomUnderSampler(random_state = seed)
            if i1 == 2:
                res = NearMiss(version = 1)
            if i1 == 3:
                res = NearMiss(version = 2)
            if i1 == 4:
                res = ClusterCentroids(random_state = seed)
            if i1 == 5:
                res = EditedNearestNeighbours()
            if i1 == 6:
                res = CondensedNearestNeighbour(random_state = seed)
            if i1 == 7:
                res = TomekLinks()
            X_re, y_re = res.fit_resample(X,y)
        else:
            if i1 == 0:
                X_re, y_re = X, y
            if i1 == 8:
                X_re, y_re = PSU(X,y)
            
        temp_color = []
        
        for i2 in range(len(y_re)):
            if y_re[i2] > 0:
                temp_color.append("chocolate")
            else:
                temp_color.append("lightsteelblue")
    
        plt.xlabel("X", fontsize = 15)
        plt.ylabel("Y", fontsize = 15)
        plt.scatter(X_re[:, 0], X_re[:, 1], edgecolors='k', c = temp_color)
        plt.xlim(min(X[:,0]) - 0.5, max(X[:,0]) + 0.5)
        plt.ylim(min(X[:,1]) - 0.5, max(X[:,1]) + 0.5)
        plt.show()

## Figure 4.
for i0 in range(4):
    
    temp1 = np.array(Dataset[2][i0][:,list([0,1,2,3,7])])
    XX = list(np.array(pd.DataFrame(-temp1).rank(method = "average", axis = 1)).ravel())
    
    temp2 = np.array(Dataset[2][-1])
    YY = list(np.array(pd.DataFrame(-temp2).rank(method = "average", axis = 1)).ravel())

    #Spearman test
    ρ, p_value = np.round(ss.spearmanr(XX,YY),4)
    
    mat = np.zeros((5,5))
    for i1 in range(len(XX)):
        if XX[i1] - int(XX[i1]) != 0:        
            t1_1 = int(XX[i1])
            t1_2 = int(XX[i1]) + 1        
            if YY[i1] - int(YY[i1]) != 0:
                t2_1 = int(YY[i1])
                t2_2 = int(YY[i1]) + 1            
                mat[t1_1 - 1, t2_1 - 1] = mat[t1_1 - 1, t2_1 - 1] + 0.25
                mat[t1_1 - 1, t2_2 - 1] = mat[t1_1 - 1, t2_2 - 1] + 0.25
                mat[t1_2 - 1, t2_1 - 1] = mat[t1_2 - 1, t2_1 - 1] + 0.25
                mat[t1_2 - 1, t2_2 - 1] = mat[t1_2 - 1, t2_2 - 1] + 0.25            
            else:
                t2 = int(YY[i1])            
                mat[t1_1 - 1, t2 - 1] = mat[t1_1 - 1, t2 - 1] + 0.5
                mat[t1_2 - 1, t2 - 1] = mat[t1_2 - 1, t2 - 1] + 0.5
        else:        
            t1 = int(XX[i1])        
            if YY[i1] - int(YY[i1]) != 0:            
                t2_1 = int(YY[i1])
                t2_2 = int(YY[i1]) + 1            
                mat[t1 - 1, t2_1 - 1] = mat[t1 - 1, t2_1 - 1] + 0.5
                mat[t1 - 1, t2_2 - 1] = mat[t1 - 1, t2_2 - 1] + 0.5
            else:            
                t2 = int(YY[i1])            
                mat[t1-1, t2-1] = mat[t1-1, t2-1] + 1
    
    ax = plt.figure().gca()
    plt.text(0.85,5.1,"ρ = " + str(ρ), fontsize = 11)
    plt.text(0.85,4.8,"p-value = " + str(p_value), fontsize = 11)
    mat = np.round(mat * 100/ np.sum(mat), 2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Classification Performance (Rank)', fontsize=15)
    plt.ylabel('Fitness Index (ϵ) (Rank)', fontsize=15)
    plt.imshow(mat, cmap = "Reds", origin = "lower", extent = [0.5,5.5,0.5,5.5])
    cbar = plt.colorbar(ticks = list([0,2,4,8,6,10,12,14,16]), fraction = 0.046, pad = 0.04)
    cbar.ax.set_yticklabels(list(["0%", "2%", "4%", "6%", "8%", "10%"]))    
    plt.show()

## Table 1.
dataset = Dataset[0]
temp_ratio = np.around([np.sum(list(dataset.values())[i][:,-1] < 0) / np.sum(list(dataset.values())[i][:,-1] > 0) for i in range(55)], decimals = 1)
temp_order = np.argsort(temp_ratio)
temp_data = {"Number of instances" : np.array([np.shape(list(dataset.values())[i])[0] for i in range(55)])[temp_order],
             "Number of attributes": np.array([np.shape(list(dataset.values())[i])[1] for i in range(55)])[temp_order],
             "Minor ratio" : np.sort(temp_ratio)}

Table_1 = pd.DataFrame(index = np.array(list(dataset.keys()))[temp_order],
                       data = temp_data)
Table_1.index.name = "dataset"
print(Table_1)

## Table 2.
data_Table2 = Dataset[2]
performance_Linear = np.around([np.mean(data_Table2[i], axis = 0) for i in [0,1]], decimals = 4)
performance_RBF = np.around([np.mean(data_Table2[i], axis = 0) for i in [2,3]], decimals = 4)
temp0 = []

for i0 in range(4):
    temp0.append(np.round(np.mean(np.array(pd.DataFrame(-Dataset[2][i0]).rank(method = "average", axis = 1)), axis = 0),2)        )

train_ratio_list = np.around(np.mean(Dataset[3], axis = 0), decimals = 2)
sampling_time = np.around(np.sum(Dataset[4], axis = 1) / 100, decimals = 2)

temp_data = {"AUC" : performance_Linear[0],
             "G-mean" : performance_Linear[1],
             "Mean Rank (AUC)" : temp0[0],
             "Mean Rank (G-mean)" : temp0[1],
             "Train Ratio (%)" : train_ratio_list,
             "Resampling Time (s)" : sampling_time}

Table_2_Linear = pd.DataFrame(index = ["RUS", "NearMiss-1", "NearMiss-2", "Cluster Centroid", "ENN", "CNN", "Tomek Links", "PSU"],
                              data = temp_data)

Table_2_Linear.index.name = "Linear Kernel"

temp_data = {"AUC" : performance_RBF[0],
             "G-mean" : performance_RBF[1],
             "Mean Rank (AUC)" : temp0[2],
             "Mean Rank (G-mean)" : temp0[3],
             "Train Ratio (%)" : train_ratio_list,
             "Resampling Time (s)" : sampling_time}

Table_2_RBF = pd.DataFrame(index = ["RUS", "NearMiss-1", "NearMiss-2", "Cluster Centroid", "ENN", "CNN", "Tomek Links", "PSU"],
                              data = temp_data)

Table_2_RBF.index.name = "RBF Kernel"

print(Table_2_Linear)
print(Table_2_RBF)


## Table 3.
temp = []
for i0 in range(4):
    temp1 = np.array(Dataset[2][i0])
    
    for i2 in range(7):
        temp.append(np.around(ss.wilcoxon(temp1[:,i2], temp1[:,-1])[1], decimals = 4))    
        
temp = np.array(temp).reshape(4,7).T

Table_3 = pd.DataFrame(index = ["RUS", "NearMiss-1", "NearMiss-2", "Cluster Centroid", "ENN", "CNN", "Tomek Links"],
                       columns = ["AUC(Linear)", "G-mean(Linear)", "AUC(RBF)", "G-mean(RBF)"],
                       data = temp)

Table_3.index.name = "Benchmark Methods"

print(Table_3)
