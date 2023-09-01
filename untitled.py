from __future__ import absolute_import, print_function, division, unicode_literals
import tensorflow as tf
import numpy as np
import math
from numpy import random,mat
import random
import gc
import torch
import pickle as pkl
import scipy.sparse as sp
import sys
import time
print(tf.__version__)
print(tf.executing_eagerly())

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
Dataset = 'dataset'
Sparse = False 
Batch_Size = 50000
Epochs = 1000
Patience = 500
Learning_Rate =0.001
Weight_Decay = 0
ffd_drop = 0
attn_drop = 0
Residual = False
dataset = Dataset

# training params
batch_size = Batch_Size
nb_epochs = Epochs
patience = Patience
lr = Learning_Rate
l2_coef = Weight_Decay
residual = Residual
hid_units = [64] # numbers of hidden units per each attention head in each layer
n_heads = [8,8] # additional entry for the output layer
nonlinearity = tf.nn.elu
optimizer = tf.keras.optimizers.Adam(learning_rate = lr)



"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
                else:
                    mt[g][i][j] = 0.0
    return -1e9 * (1.0 - mt)



###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
def load_random_data(size):

    adj = sp.random(size, size, density=0.002) # density similar to cora
    features = sp.random(size, 1000, density=0.015)
    int_labels = np.random.randint(7, size=(size))
    labels = np.zeros((size, 7)) # Nx7
    labels[np.arange(size), int_labels] = 1

    train_mask = np.zeros((size,)).astype(bool)
    train_mask[np.arange(size)[0:int(size/2)]] = 1

    val_mask = np.zeros((size,)).astype(bool)
    val_mask[np.arange(size)[int(size/2):]] = 1

    test_mask = np.zeros((size,)).astype(bool)
    test_mask[np.arange(size)[int(size/2):]] = 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
  
    # sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    # This is where I made a mistake, I used (adj.row, adj.col) instead 
    indices = np.vstack((adj.col, adj.row)).transpose()    
    
    return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)

class attn_head(tf.keras.layers.Layer):
    def __init__(self,hidden_dim, nb_nodes = None,in_drop=0.0, coef_drop=0.0,activation = tf.nn.elu,residual = False):        
        super(attn_head,self).__init__()        
        self.activation = activation
        self.residual = residual
        self.hidden_dim=hidden_dim
        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)        
        self.conv_no_bias = tf.keras.layers.Conv1D(hidden_dim,1,use_bias=False)
        self.conv_f1 = tf.keras.layers.Conv1D(1,1)
        self.conv_f2 = tf.keras.layers.Conv1D(1,1)
                
        self.conv_residual = tf.keras.layers.Conv1D(hidden_dim,1)
        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))

    def __call__(self,seq,bias_mat,training):
        seq = self.in_dropout(seq,training = training)
        # seq_fts.shape: (num_graph, num_nodes, hidden_dim)
        seq_fts = self.conv_no_bias(seq)
        # f_1.shape: (num_graph, num_nodes, 1)
        f_1 = self.conv_f1(seq_fts)
        f_2 = self.conv_f2(seq_fts)
        # logits.shape: (num_graph, num_nodes, num_nodes)
        logits = f_1 + tf.transpose(f_2,[0,2,1])
        # coefs.shape: (num_graph, num_nodes, num_nodes)
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits)*adj_t+bias_mat)
        coefs = self.coef_dropout(coefs,training = training)
        seq_fts = self.in_dropout(seq_fts,training = training)
        # vals.shape: (num_graph, num_nodes, num_nodes)
        vals = tf.matmul(coefs, seq_fts)
        vals = tf.cast(vals, dtype=tf.float32)
        ret = vals + self.bias_zero
        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq)                
            else:
                ret = ret + seq
        # shape: (num_graph, num_nodes, hidden_dim)
        return self.activation(ret)
    
    
class sp_attn_head(tf.keras.layers.Layer):
    def __init__(self,hidden_dim, nb_nodes,in_drop=0.0, coef_drop=0.0,activation = tf.nn.elu,residual = False):        
        super(sp_attn_head,self).__init__()     
        self.hidden_dim = hidden_dim
        self.nb_nodes = nb_nodes
        self.activation = activation
        self.residual = residual
        self.bn = tf.keras.layers.BatchNormalization()
        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)        
        
        self.conv_no_bias = tf.keras.layers.Conv1D(hidden_dim,1,use_bias=False)
        self.conv_f1 = tf.keras.layers.Conv1D(1,1)
        self.conv_f2 = tf.keras.layers.Conv1D(1,1)
                
        self.conv_residual = tf.keras.layers.Conv1D(hidden_dim,1)
        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))
        
    def __call__(self,seq,bias_mat,training):

        adj_mat = bias_mat
        seq = self.in_dropout(seq,training = training)
        seq_fts = self.conv_no_bias(seq)
        f_1 = self.conv_f1(seq_fts)
        f_2 = self.conv_f2(seq_fts)
        
        f_1 = tf.reshape(f_1, (self.nb_nodes, 1))
        f_1 = adj_mat*f_1
        f_2 = tf.reshape(f_2, (self.nb_nodes, 1))
        f_2 = adj_mat * tf.transpose(f_2, [1,0])
        logits = tf.compat.v1.sparse_add(f_1,f_2)


        lrelu = tf.SparseTensor(indices=logits.indices, 
                        values=tf.nn.leaky_relu(logits.values), 
                        dense_shape=logits.dense_shape)
        coefs = tf.compat.v2.sparse.softmax(lrelu)
        
        if training != False:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=self.coef_dropout(coefs.values,training = training),
                                    dense_shape=coefs.dense_shape)
            seq_fts = self.in_dropout(seq_fts,training = training)
        
        coefs = tf.compat.v2.sparse.reshape(coefs, [nb_nodes, nb_nodes])
        
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse.sparse_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, self.nb_nodes, self.hidden_dim])
        
        ret = vals + self.bias_zero
        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq)                
            else:
                ret = ret + seq
        return self.activation(ret)    
def choose_attn_head(Sparse):
    if Sparse:
        chosen_attention = sp_attn_head
    else:
        chosen_attention = attn_head
    
    return chosen_attention

class inference(tf.keras.layers.Layer):
    def __init__(self,n_heads,hid_units,nb_classes, nb_nodes,Sparse,ffd_drop=0.0, attn_drop=0.0,activation = tf.nn.elu,residual = False):        
        super(inference,self).__init__()
        attned_head = choose_attn_head(Sparse)
        self.attns = []
        self.sec_attns = []
        self.third_attns=[]
        self.final_attns = []
        self.final_sum = n_heads[-1]
        
        for i in range(n_heads[0]):
            self.attns.append(attned_head(hidden_dim = hid_units[0], nb_nodes = nb_nodes,
                                            in_drop = ffd_drop, coef_drop = attn_drop, 
                                            activation = activation,
                                            residual = residual))
        for i in range(n_heads[-1]):
            self.final_attns.append(attned_head(hidden_dim = nb_classes, nb_nodes = nb_nodes,                                                                                                         
                                                in_drop = ffd_drop, coef_drop = attn_drop, 
                                                activation = lambda x: x,
                                                residual = residual))                

    def __call__(self,inputs,bias_mat,training):        
        first_attn = []
        out = []
        for indiv_attn in self.attns:
            first_attn.append(indiv_attn(seq = inputs, bias_mat = bias_mat,training = training))
        # h_1.shape: (num_graph, num_nodes, hidden_dim*n_heads[0])
        h_1 = tf.concat(first_attn,axis = -1)
        for indiv_attn in self.final_attns:
            out.append(indiv_attn(seq=h_1,bias_mat = bias_mat,training = training))
        # logits.shape: (num_graph, num_nodes, nb_classes)
        logits = tf.add_n(out)/self.final_sum
        return logits
class GAT(tf.keras.Model):
    def __init__(self, hid_units,n_heads, nb_classes, nb_nodes,Sparse,ffd_drop = 0.0,attn_drop = 0.0,activation = tf.nn.elu,residual=False):    
        super(GAT,self).__init__()
        '''
        hid_units: 隐藏单元个数
        n_heads: 每层使用的注意力头个数
        nb_classes: 类别数，7
        nb_nodes: 节点的个数，2708
        activation: 激活函数
        residual: 是否使用残差连接
        '''                        
        self.hid_units = hid_units        
        self.n_heads = n_heads            
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.activation = activation
        self.residual = residual        
        
        self.inferencing = inference(n_heads,hid_units,nb_classes,nb_nodes,Sparse = Sparse,ffd_drop = ffd_drop,attn_drop = attn_drop, activation = activation,residual = residual)

    def micro_f1(self,logits, labels, mask):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))
        import tensorflow.compat.v1 as tf1
        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)

        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(mask, -1)
        
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf1.count_nonzero(predicted * labels * mask)
        tn = tf1.count_nonzero((predicted - 1) * (labels - 1) * mask)
        fp = tf1.count_nonzero(predicted * (labels - 1) * mask)
        fn = tf1.count_nonzero((predicted - 1) * labels * mask)

        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure

    def kl_divergence(self,preds,labels,mask):
        a=np.where(mask[0]==1)
        preds=tf.nn.softmax(preds)
        labels=tf.gather(labels, axis=1, indices=a)
        preds=tf.gather(preds, axis=1, indices=a)
        kl = tf.keras.losses.KLDivergence()
        return kl(labels, preds).numpy()
    def masked_softmax_cross_entropy(self,preds, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)


    def masked_accuracy(self,preds, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    
    def __call__(self,inputs,training,bias_mat,lbl_in,msk_in):     
        
        # logits.shape: (num_graph, num_nodes, nb_classes)         
        logits = self.inferencing(inputs = inputs, bias_mat = bias_mat,training = training) 
        log_resh = tf.reshape(logits, [-1, self.nb_classes])        
        lab_resh = tf.reshape(lbl_in, [-1, self.nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])        
        
        loss = self.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        accuracy = self.masked_accuracy(log_resh, lab_resh, msk_resh)
        
        return logits,accuracy,loss
def train(model,inputs,bias_mat,lbl_in,msk_in,
          training): 
    with tf.GradientTape() as tape:                
        logits,accuracy,loss = model(inputs = inputs,
                                     training =True,
                                     bias_mat = bias_mat,
                                     lbl_in =  lbl_in,
                                     msk_in =  msk_in)   
        
    gradients = tape.gradient(loss,model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    optimizer.apply_gradients(gradient_variables)        
                
    return logits,accuracy,loss

def evaluate(model,inputs,bias_mat,lbl_in,msk_in,training):                                                        
    logits,accuracy,loss = model(inputs= inputs,
                                     bias_mat = bias_mat,
                                     lbl_in = lbl_in,
                                     msk_in = msk_in,
                                     training = False)                        
    return logits,accuracy,loss
def predict(model,inputs,bias_mat,lbl_in,msk_in,training):                                                        
    logits,accuracy,loss = model(inputs= inputs,
                                     bias_mat = bias_mat,
                                     lbl_in = lbl_in,
                                     msk_in = msk_in,
                                     training = False)                        
    return tf.nn.softmax(logits)


#from models import GAT
#from utils import process

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))

import pickle as pkl
import scipy.sparse
import numpy as np
import pandas as pd
from scipy import sparse as sp
# import networkx as nx
from collections import defaultdict
from scipy.stats import uniform
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

#shape[0]就是读取矩阵第一维度的长度
#例如train_mask idx_train是range(M),将idx_train内的元素依次带入后赋值 返回的是布尔类型数组
train_mask = sample_mask(idx_train, new_label.shape[0])
pred_mask = sample_mask(idx_pred, new_label.shape[0])
val_mask = sample_mask(idx_val, new_label.shape[0])
test_mask = sample_mask(idx_test, new_label.shape[0])

labels_binary_train = np.zeros(new_label.shape)
labels_binary_pred = np.zeros(new_label.shape)
labels_binary_val = np.zeros(new_label.shape)
labels_binary_test = np.zeros(new_label.shape)
labels_binary_train[train_mask, :] = new_label[train_mask, :]
labels_binary_pred[pred_mask, :] = new_label[pred_mask, :]
labels_binary_val[val_mask, :] = new_label[val_mask, :]
labels_binary_test[test_mask, :] = new_label[test_mask, :]

y_train=labels_binary_train
y_pred=labels_binary_pred
y_val=labels_binary_val
y_test=labels_binary_test

features=features.todense()
adj=np.load('dataset.npy')
np.fill_diagonal(adj,1)
adj_t=tf.convert_to_tensor(adj,dtype=tf.float32)[np.newaxis]
def adj_to_bais1(adj):
    adj_bais=adj.copy()
    adj_bais[adj > 0] = 1
    adj_bais[adj <= 0] = 0
    return -1e9 * (1.0 - adj_bais)
# features, spars = preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]


features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_pred=y_pred[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
pred_mask=pred_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]


print(f'These are the parameters')
print(f'batch_size: {batch_size}')
print(f'nb_nodes: {nb_nodes}')
print(f'ft_size: {ft_size}')
print(f'nb_classes: {nb_classes}')


if Sparse:
    biases = preprocess_adj_bias(adj)
else:
    biases = adj_to_bais1(adj)[np.newaxis]
model = GAT(hid_units,n_heads, nb_classes, nb_nodes,Sparse,ffd_drop = ffd_drop,attn_drop = attn_drop,activation = tf.nn.elu,residual=False)
print('model: ' + str('SpGAT' if Sparse else 'GAT'))

vlss_mn = np.inf
vacc_mx = 0.0
curr_step = 0
train_loss_avg = 0
train_acc_avg = 0
val_loss_avg = 0
val_acc_avg = 0
model_number = 0
import math
import keras
# 继承自定义学习率的类
class CosineWarmupDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''
    initial_lr: 初始的学习率
    min_lr: 学习率的最小值
    max_lr: 学习率的最大值
    warmup_step: 线性上升部分需要的step
    total_step: 第一个余弦退火周期需要对总step
    multi: 下个周期相比于上个周期调整的倍率
    print_step: 多少个step并打印一次学习率
    '''
    # 初始化
    def __init__(self, initial_lr, min_lr, warmup_step, total_step, multi, print_step):
        # 继承父类的初始化方法
        super(CosineWarmupDecay, self).__init__()
        
        # 属性分配
        self.initial_lr = tf.cast(initial_lr, dtype=tf.float32)
        self.min_lr = tf.cast(min_lr, dtype=tf.float32)
        self.warmup_step = warmup_step  # 初始为第一个周期的线性段的step
        self.total_step = total_step    # 初始为第一个周期的总step
        self.multi = multi
        self.print_step = print_step
        
        # 保存每一个step的学习率
        self.learning_rate_list = []
        # 当前步长
        self.step = 0
        
        
    # 前向传播, 训练时传入当前step，但是上面已经定义了一个，这个step用不上
    def __call__(self, step):
        
        # 如果当前step达到了当前周期末端就调整
        if  self.step>=self.total_step:
            
            # 乘上倍率因子后会有小数，这里要注意
            # 调整一个周期中线性部分的step长度
            self.warmup_step = self.warmup_step * (1 + self.multi)
            # 调整一个周期的总step长度
            self.total_step = self.total_step * (1 + self.multi)
            
            # 重置step，从线性部分重新开始
            self.step = 0
            
        # 余弦部分的计算公式
        decayed_learning_rate = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) *       \
                                (1 + tf.math.cos(math.pi * (self.step-self.warmup_step) /        \
                                  (self.total_step-self.warmup_step)))
        
        # 计算线性上升部分的增长系数k
        k = (self.initial_lr - self.min_lr) / self.warmup_step 
        # 线性增长线段 y=kx+b
        warmup = k * self.step + self.min_lr
        
        # 以学习率峰值点横坐标为界，左侧是线性上升，右侧是余弦下降
        decayed_learning_rate = tf.where(self.step<self.warmup_step, warmup, decayed_learning_rate)
        
        
        # 每个epoch打印一次学习率
        if step % self.print_step == 0:
            # 打印当前step的学习率
            print('learning_rate has changed to: ', decayed_learning_rate.numpy().item())
        
        # 每个step保存一次学习率
        self.learning_rate_list.append(decayed_learning_rate.numpy().item())
 
        # 计算完当前学习率后step加一用于下一次
        self.step = self.step + 1
        
        # 返回调整后的学习率
        return decayed_learning_rate
# 迭代次数
num_epochs = 2000
# 初始学习率
initial_lr = 0.01
# 学习率下降的最小值
min_lr = 1e-6
# 余弦退火的周期调整倍率
multi = 0.25
# 一个epoch包含多少个batch也是多少个steps, 即1875
one_epoch_batchs = 1
 
# 第一个余弦退火周期需要的总step，以三个epoch为一个周期
total_step = 20
 
# 线性上升部分需要的step, 一个周期的四分之一的epoch用于线性上升
warmup_step = int(total_step * 0.25)
 
# 多少个step打印一次学习率, 一个epoch打印一次
print_step = one_epoch_batchs

cosinewarmupdecay = CosineWarmupDecay(initial_lr=initial_lr, # 初始学习率，即最大学习率
                                  min_lr=min_lr,             # 学习率下降的最小值
                                  warmup_step=warmup_step,   # 线性上升部分的step
                                  total_step=total_step,     # 训练的总step
                                  multi=multi,               # 周期调整的倍率
                                  print_step=print_step)     # 每个epoch打印一次学习率值
optimizer = keras.optimizers.Adam(cosinewarmupdecay)
for epoch in range(nb_epochs):
    ###Training Segment###
    tr_step = 0
    tr_size = features.shape[0]
    while tr_step * batch_size < tr_size:            
        if Sparse:
            bbias = biases
        else:
            bbias = biases[tr_step*batch_size:(tr_step+1)*batch_size]
       
        _, acc_tr,loss_value_tr = train(model,
                                        inputs=     features[tr_step*batch_size:(tr_step+1)*batch_size],
                                        bias_mat=     bbias,
                                        lbl_in =     y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                                        msk_in =train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                                        training=True)
        train_loss_avg += loss_value_tr
        train_acc_avg += acc_tr
        tr_step += 1
    ###Validation Segment###
    vl_step = 0
    vl_size = features.shape[0]
    while vl_step * batch_size < vl_size:
        
        if Sparse:
            bbias = biases
        else:
            bbias = biases[vl_step*batch_size:(vl_step+1)*batch_size]
        
        _, acc_vl,loss_value_vl = evaluate(model,
                                            inputs=     features[vl_step*batch_size:(vl_step+1)*batch_size],
                                            bias_mat=     bbias,
                                            lbl_in =     y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                                            msk_in =val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                                            training=False)
        val_loss_avg += loss_value_vl
        val_acc_avg += acc_vl
        vl_step += 1
    print('epoch:%d'%epoch)
    print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                                        (train_loss_avg/tr_step, train_acc_avg/tr_step,
                                        val_loss_avg/vl_step, val_acc_avg/vl_step))
    

    ###Early Stopping Segment###
    
    if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
            if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step            
                    working_weights = model.get_weights()
            vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
            vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
            curr_step = 0
    else:
            curr_step += 1
            if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    model.set_weights(working_weights)
                    break

    train_loss_avg = 0
    train_acc_avg = 0
    val_loss_avg = 0
    val_acc_avg = 0

ts_step = 0
ts_size = features.shape[0]
ts_loss = 0.0
ts_acc = 0.0
while ts_step * batch_size < ts_size:
    
    if Sparse:
            bbias = biases
    else:
            bbias = biases[ts_step*batch_size:(ts_step+1)*batch_size]
    
    logists1 = predict(model,
                    inputs=     features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_mat=     bbias,
                    lbl_in =     y_train[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in =train_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    training=False)

    ts_step += 1
p=logists1.numpy()[0][pred_mask[0]]
q=labels[pred_mask[0]]
csv_file = "./Infor_Data/ST_label/ST_label_2.csv"
df_original = pd.read_csv(csv_file, index_col=0)
column_names = df_original.columns
pd.DataFrame(p,columns=column_names).to_csv("p.csv",index=None)