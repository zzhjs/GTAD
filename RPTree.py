# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:46:56 2021

@author: Mashaan
"""

import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6


from sklearn import random_projection
from sklearn.decomposition import PCA
import random
class Node(object):
    def __init__(self, data):
        self.data           = data
        self.index          = None
        self.hyperplane     = None
        self.PCAmean        = None
        self.splitDimension = None
        self.splitPoint     = None
        self.left           = None
        self.right          = None

    def get_leaf_nodes(self):
        leaves = []
        self._collect_leaf_nodes(self,leaves)
        return leaves

    def _collect_leaf_nodes(self, node, leaves):
        if node is not None:
            if node.left==None and node.right==None:
                leaves.append(node.index)
                
            self._collect_leaf_nodes(node.left, leaves)
            self._collect_leaf_nodes(node.right, leaves)

class BinaryTree(object):
    def __init__(self, root):
        self.root = Node(root)
      
    def construct_tree(self, tree, index):
        nTry = 3 #100 #1000
        whichProjection = '2019_Yan' # '2008_Dasgupta' # '2019_Yan' #  'proposed' # 'PCA'
        
        X_data = tree.root.data
        if whichProjection == '2008_Dasgupta':
            transformer = random_projection.GaussianRandomProjection(n_components=X_data.shape[1]-1)            
            X_proj = transformer.fit_transform(X_data)
            hyperplane = transformer.components_
        elif whichProjection == '2019_Yan':
            dispersion = 0
            for r in range(nTry):
                # transformer = random_projection.GaussianRandomProjection(n_components=X_data.shape[1]-1,random_state=123)
                transformer = random_projection.GaussianRandomProjection(n_components=X_data.shape[1]-1)

                X_proj_temp = transformer.fit_transform(X_data)
                dispersionCurrent = np.max(np.std(X_proj_temp, axis=0))
                # print('dispersionCurrent = ' + str(dispersionCurrent))
                if  dispersionCurrent > dispersion:
                    dispersion = dispersionCurrent
                    # print('dispersion = ' + str(dispersion))
                    X_proj = X_proj_temp
                    hyperplane = transformer.components_
            # print('=============================================================')
        elif whichProjection == 'proposed':
            dispersion = 0
            for r in range(nTry):
                transformer = random_projection.GaussianRandomProjection(n_components=X_data.shape[1]-1)
                X_proj_temp = transformer.fit_transform(X_data)
                dispersionCurrent = np.max(np.std(X_proj_temp, axis=0))
                # print('dispersionCurrent = ' + str(dispersionCurrent))
                if  dispersionCurrent > dispersion:
                    dispersion = dispersionCurrent
                    # print('dispersion = ' + str(dispersion))
                    X_proj = X_proj_temp
                    hyperplane = transformer.components_
            for r in range(nTry):
                hyperplane_temp = hyperplane + np.random.normal(0, 0.1, hyperplane.shape)
                X_proj_temp = np.dot(X_data,np.transpose(hyperplane_temp))
                dispersionCurrent = np.max(np.std(X_proj_temp, axis=0))
                # print('dispersionCurrent = ' + str(dispersionCurrent))
                if  dispersionCurrent > dispersion:
                    dispersion = dispersionCurrent
                    # print('dispersion = ' + str(dispersion))
                    X_proj = X_proj_temp
                    hyperplane = hyperplane_temp
            for r in range(nTry):
                hyperplane_temp = hyperplane + np.random.normal(0, 0.01, hyperplane.shape)
                X_proj_temp = np.dot(X_data,np.transpose(hyperplane_temp))
                dispersionCurrent = np.max(np.std(X_proj_temp, axis=0))
                # print('dispersionCurrent = ' + str(dispersionCurrent))
                if  dispersionCurrent > dispersion:
                    dispersion = dispersionCurrent
                    # print('dispersion = ' + str(dispersion))
                    X_proj = X_proj_temp
                    hyperplane = hyperplane_temp
            # print('=============================================================')
        elif whichProjection == 'PCA':
            # this line could throw an error if the number of samples is less than the number of principle components selected
            pca = PCA(n_components=min(X_data.shape[0]-1, X_data.shape[1]-1))  
            pca.fit(X_data)  
            X_proj = pca.transform(X_data)
            hyperplane = pca.components_
            tree.root.PCAmean = pca.mean_

        SplitDimension = np.argmax(np.std(X_proj, axis=0))
        SplitPoint = random.uniform(np.quantile(X_proj[:,SplitDimension], 0.25, axis=0),np.quantile(X_proj[:,SplitDimension], 0.75, axis=0))
        
        X_left = X_data[np.where(X_proj[:,SplitDimension] < SplitPoint)[0]]
        X_left_index = index[np.where(X_proj[:, SplitDimension] < SplitPoint)[0]]
        X_right = X_data[np.where(X_proj[:,SplitDimension] >= SplitPoint)[0]]
        X_right_index = index[np.where(X_proj[:, SplitDimension] >= SplitPoint)[0]]

        tree.root.hyperplane = hyperplane
        tree.root.splitDimension = SplitDimension
        tree.root.splitPoint = SplitPoint
        
        # fig=plt.figure(figsize=(6,6))
        # ax=fig.add_subplot(111) 
        # ax.scatter(X_left[:,0], X_left[:,1], c='b')
        # ax.scatter(X_right[:,0], X_right[:,1], c='r')
        # plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
        #         labelbottom=False,labeltop=False,labelleft=False,labelright=False)
        
        if X_left.shape[0]>20:
            tree.root.left = self.construct_tree(BinaryTree(X_left), X_left_index)
        else:
            tree.root.left = Node(X_left)
            tree.root.left.index = X_left_index
            
        if X_right.shape[0]>20:
            tree.root.right = self.construct_tree(BinaryTree(X_right), X_right_index)
        else:
            tree.root.right = Node(X_right)
            tree.root.right.index = X_right_index
            
        return tree.root                                    
            
    def preorder_search(self, NodeRoot, NodeSearch):
        if NodeRoot.left==None or NodeRoot.right==None:
            return NodeRoot.data
        else:
            # fig=plt.figure(figsize=(6,6))
            # ax=fig.add_subplot(111) 
            # ax.scatter(NodeRoot.left.data[:,0], NodeRoot.left.data[:,1], c='b')
            # ax.scatter(NodeRoot.right.data[:,0], NodeRoot.right.data[:,1], c='r')
            # ax.scatter(NodeSearch.data[0], NodeSearch.data[1], marker='x', c='k')
            # plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
            #         labelbottom=False,labeltop=False,labelleft=False,labelright=False)
            
            if NodeRoot.PCAmean is not None:
                ProjectedPoint = np.dot((NodeSearch.data - NodeRoot.PCAmean),np.transpose(NodeRoot.hyperplane))
            else:
                ProjectedPoint = np.dot(NodeSearch.data,np.transpose(NodeRoot.hyperplane))
                
                
            # print('NodeRoot.data.shape = ' + str(NodeRoot.data.shape))
            # print('ProjectedPoint = ' + str(ProjectedPoint))
            # print('NodeRoot.splitDimension = ' + str(NodeRoot.splitDimension))
            # print('ProjectedPoint[NodeRoot.splitDimension] = ' + str(ProjectedPoint[NodeRoot.splitDimension]))
            # print('NodeRoot.splitPoint = ' + str(NodeRoot.splitPoint))                        
            if ProjectedPoint[NodeRoot.splitDimension] < NodeRoot.splitPoint:
                # print('Im going left')
                # print('=============================================================')
                return self.preorder_search(NodeRoot.left, NodeSearch)
            else:
                # print('Im going right')
                # print('=============================================================')
                return self.preorder_search(NodeRoot.right, NodeSearch)    
            
            
            
