import glob
import pandas as pd
import os
import itertools
import numpy as np
import pandas as pd
import random
from sklearn.neighbors import KDTree
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from warnings import simplefilter
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.sparse
from scipy import sparse as sp
import networkx as nx
from collections import defaultdict
from scipy.stats import uniform
simplefilter(action='ignore', category=FutureWarning)

outputdir='Infor_Data'
path0 = os.path.join(os.getcwd(), outputdir)
#' import processed data
files1 = glob.glob(path0 + "/ST_count/*.csv")
files1.sort()
count_list = []
for df in files1:
    print(df)
    count_list.append(pd.read_csv(df, index_col=0))

files2 = glob.glob(path0 + "/ST_norm/*.csv")
files2.sort()
norm_list = []
for df in files2:
    print(df)
    norm_list.append(pd.read_csv(df, index_col=0))

files3 = glob.glob(path0 + "/ST_scale/*.csv")
files3.sort()
scale_list = []
for df in files3:
    print(df)
    scale_list.append(pd.read_csv(df, index_col=0))

files4 = glob.glob(path0 + "/ST_label/*.csv")
files4.sort()
label_list = []
for df in files4:
    print(df)
    label_list.append(pd.read_csv(df, index_col=0))

fpath = os.path.join(path0, 'Variable_features.csv')
features = pd.read_csv(fpath, index_col=False)
features=features.values.flatten()

cell_embedding = pd.read_csv(path0+"/integrated.mat.csv", index_col=0).T
def l2norm(mat):
    stat = np.sqrt(np.sum(mat**2, axis=1))
    cols = mat.columns
    mat[cols] = mat[cols].div(stat, axis=0)
    mat[np.isinf(mat)] = 0
    return mat

counts1 = count_list[0]
counts2 = count_list[1]
norm_data1 = norm_list[0]
norm_data2 = norm_list[1]
scale_data1 = scale_list[0]
scale_data2 = scale_list[1]
rowname = scale_data2.index
norm_embedding = l2norm(mat=cell_embedding)
spots1 = scale_data1.columns
spots2 = scale_data2.columns
embedding_spots1 = norm_embedding.loc[spots1, ]
embedding_spots2 = norm_embedding.tail(len(spots2))

label1 = label_list[0]
label1.to_csv('./Datadir/Pseudo_Label1.csv', index=False)
label2 = label_list[1]
label2.to_csv('./Datadir/Real_Label2.csv', index=False)
DataPath1 = '{}/Pseudo_ST1.csv'.format("Datadir")
DataPath2 = '{}/Real_ST2.csv'.format("Datadir")
LabelsPath1 = '{}/Pseudo_Label1.csv'.format("Datadir")
LabelsPath2 = '{}/Real_Label2.csv'.format("Datadir")

data1=embedding_spots1
data2=embedding_spots2
lab_label1 = pd.read_csv(LabelsPath1, header=0, index_col=False, sep=',')
lab_label2 = pd.read_csv(LabelsPath2, header=0, index_col=False, sep=',')
lab_data1 = data1.reset_index(drop=True)  #.transpose()
lab_data2 = data2.reset_index(drop=True)  #.transpose()

lab_label1= lab_label1.reset_index(drop=True)

random.seed(123)
p_data = lab_data1
p_label = lab_label1

#假ST 划分训练集和测试集
temD_train, temd_test, temL_train, teml_test = train_test_split(
    p_data, p_label, test_size=0.1, random_state=1)
temd_train, temd_val, teml_train, teml_val = train_test_split(
    temD_train, temL_train, test_size=0.1, random_state=1)

print((temd_train.index == teml_train.index).all())
print((temd_test.index == teml_test.index).all())
print((temd_val.index == teml_val.index).all())
data_train = temd_train
label_train = teml_train
data_test = temd_test
label_test = teml_test
data_val = temd_val
label_val = teml_val

data_train1 = data_train
data_test1 = data_test
data_val1 = data_val
label_train1 = label_train
label_test1 = label_test
label_val1 = label_val

train2 = pd.concat([data_train1, lab_data2])
lab_train2 = pd.concat([label_train1, lab_label2])
#' save objects
#假ST 训练集 测试机 验证集 和 real原始数据集
PIK = "{}/dataset.dat".format("Datadir")
res = [
    data_train1, data_test1, data_val1, label_train1, label_test1,
    label_val1, lab_data2, lab_label2
]
with open(PIK, "wb") as f:
    pkl.dump(res, f)
print('load data succesfully....')



PIK = "{}/dataset.dat".format("Datadir")
with open(PIK, "rb") as f:
    objects = pkl.load(f)

data_train1, data_test1, data_val1, label_train1, label_test1, label_val1, lab_data2, lab_label2 = tuple(
    objects)

#拼接P训练集和T数据集
train2 = pd.concat([data_train1, lab_data2])
#拼接P标签训练集和T标签数据集
lab_train2 = pd.concat([label_train1, lab_label2])

#创建数组对象
datas_train = np.array(train2)
datas_test = np.array(data_test1)
datas_val = np.array(data_val1)
labels_train = np.array(lab_train2)
labels_test = np.array(label_test1)
labels_val = np.array(label_val1)

#' convert pandas data frame to csr_matrix format
#将st数据数组转化为csr格式稀疏矩阵存储格式
datas_tr = scipy.sparse.csr_matrix(datas_train.astype('float64'))
datas_va = scipy.sparse.csr_matrix(datas_val.astype('float64'))
datas_te = scipy.sparse.csr_matrix(datas_test.astype('float64'))

#M是p数据训练集的长度
M = len(data_train1)

#' 4) get the feature object by combining training, test, valiation sets
#vstack 垂直堆叠稀疏矩阵（按行）  tolil 将这个矩阵转换为列表中的列表格式。
features = sp.vstack((sp.vstack((datas_tr, datas_va)), datas_te)).tolil()
"""Row-normalize feature matrix and convert to tuple representation"""

def sample_mask(idx, l):
    """Create mask."""
    #np.zeros 返回来一个给定形状和类型的用0填充的数组
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)

labels_tr = labels_train
labels_va = labels_val
labels_te = labels_test

#合并数组
labels = np.concatenate(
    [np.concatenate([labels_tr, labels_va]), labels_te])
Labels = pd.DataFrame(labels)
true_label = Labels
#' new label with binary values
new_label = labels
#从0到M
idx_train = range(M)
#从M到len(labels_tr)
idx_pred = range(M, len(labels_tr))
#从len(labels_tr)到len(labels_tr) + len(labels_va)
idx_val = range(len(labels_tr), len(labels_tr) + len(labels_va))
# 从len(labels_tr) + len(labels_va) + len(labels_tr) + len(labels_va) + len(labels_te))
idx_test = range(
    len(labels_tr) + len(labels_va),
    len(labels_tr) + len(labels_va) + len(labels_te))
features=features.todense()

import RPTree
NumOfTrees = 10
adj = sp.coo_matrix(np.zeros((features.shape[0], features.shape[0]), dtype=np.float32))
for r in range(NumOfTrees):
    tree = RPTree.BinaryTree(features)
    features_index = np.arange(features.shape[0])
    tree_root = tree.construct_tree(tree, features_index)
    # get the indices of points in leaves
    leaves_array = tree_root.get_leaf_nodes()

    # connect points in the same leaf node
    edgeList = []
    for i in range(len(leaves_array)):
        x = leaves_array[i]
        n = x.size
        perm = np.empty((n, n, 2), dtype=x.dtype)
        perm[..., 0] = x[:, None]
        perm[..., 1] = x
        perm1 = np.reshape(perm, (-1, 2))
        if i == 0:
            edgeList = perm1
        else:
            edgeList = np.vstack((edgeList, perm1))

    # assign one as edge weight
    edgeList = edgeList[edgeList[:, 0] != edgeList[:, 1]]
    edgeList = np.hstack((edgeList, np.ones((edgeList.shape[0], 1), dtype=int)))

    # convert edges list to adjacency matrix
    shape = tuple(edgeList.max(axis=0)[:2] + 1)
    adjMatRPTree = sp.coo_matrix((edgeList[:, 2], (edgeList[:, 0], edgeList[:, 1])), shape=shape,
                                     dtype=edgeList.dtype)

    # an adjacency matrix holding weights accumulated from all rpTrees
    adj = adj + (adjMatRPTree / NumOfTrees)
    print(r)
adj=adj.todense()
np.save('dataset.npy',adj)