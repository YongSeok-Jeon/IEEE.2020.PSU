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
import category_encoders as ce
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
    
    temp1 = []
    for ii in range(55):
        temp1.append(np.mean(np.array(Dataset[2][i0].ravel()).reshape(55,100,8)[ii], axis = 0))
    temp1 = np.array(temp1)
    
    XX = list(np.array(pd.DataFrame(-temp1[:,list([0,1,2,3,7])]).rank(method = "average", axis = 1)).ravel())
    
    temp2 = []
    for ii in range(55):
        temp2.append(np.mean(np.array(Dataset[2][-1].ravel()).reshape(55,100,5)[ii], axis = 0))
    temp2 = np.array(temp2)

    YY = list(np.array(pd.DataFrame(-temp2).rank(method = "average", axis = 1)).ravel())

    #Spearman test
    ρ, p_value = np.round(ss.spearmanr(XX,YY),4)
    
    mat = np.zeros((5,5))
    for i1 in range(len(XX)):
        mat[int(XX[i1]-1), int(YY[i1]-1)] = mat[int(XX[i1]-1), int(YY[i1]-1)] + 1

    #print(mat.T)
    ax = plt.figure().gca()
    plt.text(0.85,5.1,"ρ = " + str(ρ), fontsize = 11)
    plt.text(0.85,4.8,"p-value = " + str(p_value), fontsize = 11)
    mat = np.round(mat * 100/ np.sum(mat), 2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Classification Performance (Rank)', fontsize=15)
    plt.ylabel('Fitness Index (ϵ) (Rank)', fontsize=15)
    plt.imshow(mat, cmap = "Reds", origin = "lower", extent = [0.5,5.5,0.5,5.5])
    cbar = plt.colorbar(ticks = list([0,2,4,8,6,10,12,14,16]), fraction = 0.046, pad = 0.04)
    cbar.ax.set_yticklabels(list(["0%", "2%", "4%", "6%", "8%", "10%", "12%", "14%", "16%"]))    
    
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
    temp1 = []
    for i1 in range(55):
        temp1.append(np.mean(np.array(Dataset[2][i0].ravel()).reshape(55,100,8)[i1], axis = 0))
    temp1 = np.array(temp1)
    temp0.append(np.round(np.mean(np.array(pd.DataFrame(-temp1).rank(method = "average", axis = 1)), axis = 0),2)        )


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
    temp0 = []
    for i1 in range(55):
        temp0.append(np.mean(np.array(Dataset[2][i0].ravel()).reshape(55,100,8)[i1], axis = 0))
    temp0 = np.array(temp0)
    for i2 in range(7):
        temp.append(np.around(ss.wilcoxon(temp0[:,i2], temp0[:,-1])[1], decimals = 4))    
        
temp = np.array(temp).reshape(4,7).T

Table_3 = pd.DataFrame(index = ["RUS", "NearMiss-1", "NearMiss-2", "Cluster Centroid", "ENN", "CNN", "Tomek Links"],
                       columns = ["AUC(Linear)", "G-mean(Linear)", "AUC(RBF)", "G-mean(RBF)"],
                       data = temp)

Table_3.index.name = "Benchmark Methods"

print(Table_3)
    
################################################################################################################################################
### Experiment
################################################################################################################################################

# paramters
C_list = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
gamma_list = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]


def Resample(aa,bb,cc,dd):
    
    seed = dd

    if cc == 8:
        
        X_res, y_res = PSU(aa, bb)

    if cc not in [8]:

        if cc == 1:
            res = RandomUnderSampler(random_state = seed)
        if cc == 2:
            res = NearMiss(version = 1)
        if cc == 3:
            res = NearMiss(version = 2)
        if cc == 4:
            res = ClusterCentroids(random_state=seed)
        if cc == 5:
            res = EditedNearestNeighbours()
        if cc == 6:
            res = CondensedNearestNeighbour(random_state = seed)
        if cc == 7:
            res = TomekLinks()
            
        y_temp = []
    
        for i in range(len(bb)):
            if bb[i] == -1:
                y_temp.append(0)
            else:
                y_temp.append(1)

        y_temp = np.asarray(y_temp)
        
        X_res, y_res = res.fit_resample(aa, y_temp)
        
        for i in range(len(y_res)):
            if y_res[i] == 0:
                y_res[i] = -1    

    return [X_res, y_res]


def test(seed, resample, measure, kernel):

    score = []
    
    for i0 in range(55):
        data = list(Dataset[0].values())[i0]
        
        X = data[:,:-1]
        y = data[:,-1]
        y = y.astype('int')
        
        skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle = True)
        
        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

        temp_score0 = []
        temp_par = []
        
        X_list = []
        X_list2 = []
        y_list = []
        y_list2 = []
        
        # 5-fold split
        for train_index, test_index in skf.split(X_train,y_train):

            temp_train_X = X_train[train_index]
            temp_test_X  = X_train[test_index]
            
            le =  ce.OneHotEncoder()
            le.fit(np.vstack((temp_train_X, temp_test_X)))

            temp_train_X_onehot = np.asarray(le.transform(temp_train_X))
            temp_test_X_onehot = np.asarray(le.transform(temp_test_X))

            sc = MinMaxScaler()
            temp_train_X_onehot_scale = sc.fit_transform(temp_train_X_onehot)
            temp_test_X_onehot_scale = sc.transform(temp_test_X_onehot)

            train_X, train_y = Resample(temp_train_X_onehot_scale, y_train[train_index], resample, seed)
                        
            X_list.append(train_X)
            y_list.append(train_y)

            
            X_list2.append(temp_test_X_onehot_scale)
            y_list2.append(y_train[test_index])
            
        
        # linear kernel
        if kernel == "linear":
            for i1 in C_list:
                temp_score1 = []
                
                for i2 in range(5):
                    train_X = X_list[i2]
                    train_y = y_list[i2]
                    test_X = X_list2[i2]                    
                    test_y = y_list2[i2]
                    
                    clf = SVC(C = i1, kernel = "linear", max_iter = 10000)
                        
                    clf.fit(train_X, train_y)
                    y_pred = clf.predict(test_X)
                    mat = confusion_matrix(test_y,y_pred)
                    
                    if np.shape(mat) == (2,2):
                        if measure == "AUC":
                            temp_score1.append(0.5 * (mat[0,0] / sum(mat[0,:]) + mat[1,1] / sum(mat[1,:])))
                        else:
                            temp_score1.append(((mat[0,0] / sum(mat[0,:])) * (mat[1,1] / sum(mat[1,:])))**0.5)

                    if np.shape(mat) == (1,1):
                        
                        if measure == "AUC":
                            temp_score1.append(0.5)
                        else:
                            temp_score1.append(0)
                        

                if measure == "AUC":
                    temp_score0.append(sum(temp_score1)/len(temp_score1))
                    temp_par.append(i1)
                else:
                    temp_score0.append(sum(temp_score1)/len(temp_score1))
                    temp_par.append(i1)
        
        
        # rbf kernel
        if kernel == "rbf":
            for i1 in C_list:
                for i2 in gamma_list:
                    temp_score1 = []
                
                    for i3 in range(5):
                        train_X = X_list[i3]
                        train_y = y_list[i3]
                        test_X = X_list2[i3]                    
                        test_y = y_list2[i3]
                    
                        clf = SVC(C = i1, kernel = "rbf", gamma = i2, max_iter = 10000)
            
                        clf.fit(train_X, train_y)
                    
                        y_pred = clf.predict(test_X)
                        mat = confusion_matrix(test_y,y_pred)
                        
                        if np.shape(mat) == (2,2):
                            if measure == "AUC":
                                temp_score1.append(0.5 * (mat[0,0] / sum(mat[0,:]) + mat[1,1] / sum(mat[1,:])))
                            else:
                                temp_score1.append(((mat[0,0] / sum(mat[0,:])) * (mat[1,1] / sum(mat[1,:])))**0.5)
                                
                        if np.shape(mat) == (1,1):
                            if measure == "AUC":
                                temp_score1.append(0.5)
                            else:
                                temp_score1.append(0)
                        
                    if measure == "AUC":
                        temp_score0.append(sum(temp_score1)/len(temp_score1))
                        temp_par.append([i1,i2])
                    else:
                        temp_score0.append(sum(temp_score1)/len(temp_score1))
                        temp_par.append([i1,i2])
                    

        temp_where = temp_score0.index(max(temp_score0))
        
        if kernel == "linear":
            clf = SVC(C = temp_par[temp_where], kernel = "linear", max_iter = 10000)
        
        if kernel == "rbf":
            clf = SVC(C = temp_par[temp_where][0], kernel = "rbf", gamma = temp_par[temp_where][1], max_iter = 10000)

        le =  ce.OneHotEncoder()
        le.fit(np.vstack((X_train, X_test)))
        
        X_train_onehot = np.asarray(le.transform(X_train))
        X_test_onehot = np.asarray(le.transform(X_test))

        sc = MinMaxScaler()
        X_train_onehot_scale = sc.fit_transform(X_train_onehot)
        X_test_onehot_scale = sc.transform(X_test_onehot)

        train_X, train_y = Resample(X_train_onehot_scale,y_train,resample,seed)
        train_y = train_y.ravel()
        
        clf.fit(train_X, train_y)
        
        y_test_hat = clf.predict(X_test_onehot_scale)
        mat = confusion_matrix(y_test,y_test_hat)
        
        if np.shape(mat) == (2,2):
            if measure == "AUC":
                score.append(0.5 * (mat[0,0] / sum(mat[0,:]) + mat[1,1] / sum(mat[1,:])))
            else:
                score.append(((mat[0,0] / sum(mat[0,:])) * (mat[1,1] / sum(mat[1,:])))**0.5)
        if np.shape(mat) == (1,1):
            
            if measure == "AUC":
                score.append(0.5)
            else:
                score.append(0)
                
    return score

for i0 in range(1,9):
    for i1 in range(1,101):
        model = test(i1,i0,"AUC", "linear")
        #model = test(i1,i0,"G-Mean", "linear")
        #model = test(i1,i0,"AUC", "rbf")
        #model = test(i1,i0,"G-Mean", "rbf")
        
        
    
