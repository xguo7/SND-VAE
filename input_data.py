import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.io as io
import open3d
import pickle
import json
import copy

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def scipy_spanning_tree(edge_index, num_nodes, num_edges):
    row, col = edge_index[:,0], edge_index[:,1]
    cgraph = csr_matrix((np.random.random(num_edges) + 1, (row, col)), shape=(num_nodes, num_nodes))
    Tcsr = minimum_spanning_tree(cgraph)
    tree_row, tree_col = Tcsr.nonzero()
    spanning_edges = np.concatenate([[tree_row], [tree_col]]).T
    return spanning_edges

def build_spanning_tree_edge(edge_index, algo='union', num_nodes=None, num_edges=None):
    # spanning_edges
    if algo=='union':
        spanning_edges = random_spanning_tree(edge_index)
    elif algo=='scipy':
        spanning_edges = scipy_spanning_tree(edge_index, num_nodes, num_edges)
    
    spanning_edges = spanning_edges.T
    spanning_edges_undirected = np.array([
            np.concatenate([spanning_edges[0], spanning_edges[1]]),
            np.concatenate([spanning_edges[1], spanning_edges[0]]),
        ])
    return spanning_edges_undirected

def generate_adj_3d(adj, batch_index):
    #this is to generate the mutual distance (relation attributes) for each pair of nodes
    adj_3d=[]
    for adj_ in adj:
        g_3d=np.zeros((len(adj_), len(adj_), len(adj_)))
        for i in range(len(adj_)):
            for j in range(len(adj_)):
                for k in range(len(adj_)):
                    if adj_[i,j]==1 and adj_[j,k]==1:
                        g_3d[i,j,k]=1
        adj_3d.append(g_3d)
    np.save('2D_adj_3d'+'_'+str(batch_index)+'.npy',adj_3d)
    return adj_3d

def load_data_syn(type_, path):
    if type_ == 'train':
        adj=np.load(path+'/train/2D_adj.npy')
        node=np.load(path+'/train/2D_node.npy')/120
        spatial=np.load(path+'/train/2D_geometry.npy')/600
        rel=np.load(path+'/train/2D_rel.npy')/600
        factor=np.load(path+'/train/2D_prop.npy')
        new_adj=[]
        for n in range(len(adj)):
            new_adj.append(adj[n].toarray())
            for i in range(len(new_adj[n])):
                new_adj[n][i,i]=0
                for j in range(len(new_adj[n])):
                    assert new_adj[n][i,j]==new_adj[n][j,i]
        new_adj = np.array(new_adj)
        print (new_adj.shape)
        new_new_adj = []
        for adj in new_adj:
            x, y = np.where(adj)
            # print (x.shape, y.shape)
            edges = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
            new_new_adj_sub = []
            raw_edges = copy.deepcopy(edges)
            for i in range(FLAGS.sampling_num):
                edges = build_spanning_tree_edge(raw_edges, 'scipy', node.shape[1], len(raw_edges))
                adj = np.zeros_like(adj)
                adj[edges[0],edges[1]] = 1
                new_new_adj_sub.append(adj)
            new_new_adj.append(new_new_adj_sub)
        adj = np.array(new_new_adj)
        
        index = [i for i in range(len(node))]   #randomly shuffle the dataset
        np.random.shuffle(index)
        new_adj = new_adj[index]
        adj = adj[index]
        node = node[index]
        spatial = spatial[index]
        rel=rel[index]
        factor=factor[index]

        print (node.shape, adj.shape, new_adj.shape)

        return  node, spatial, adj, rel, factor, np.array(new_adj)

    if type_ in ['test_generation','test_disentangle','test_reconstruct', 'test']:
        adj=np.load(path+'/test/2D_adj.npy')
        node=np.load(path+'/test/2D_node.npy')/120
        spatial=np.load(path+'/test/2D_geometry.npy') /600
        rel=np.load(path+'/test/2D_rel.npy')/600
        factor=np.load(path+'/train/2D_prop.npy')
        new_adj=[]
        for n in range(len(adj)):
            new_adj.append(adj[n].toarray())
            for i in range(len(new_adj[n])):
                new_adj[n][i,i]=0
                for j in range(len(new_adj[n])):
                    assert new_adj[n][i,j]==new_adj[n][j,i]
        new_adj = np.array(new_adj)
        print (new_adj.shape)
        new_new_adj = []
        for adj in new_adj:
            x, y = np.where(adj)
            # print (x.shape, y.shape)
            edges = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
            new_new_adj_sub = []
            raw_edges = copy.deepcopy(edges)
            for i in range(FLAGS.sampling_num):
                edges = build_spanning_tree_edge(raw_edges, 'scipy', node.shape[1], len(raw_edges))
                adj = np.zeros_like(adj)
                adj[edges[0],edges[1]] = 1
                new_new_adj_sub.append(adj)
            new_new_adj.append(new_new_adj_sub)
        adj = np.array(new_new_adj)
        
        index = [i for i in range(len(node))]   #randomly shuffle the dataset
        np.random.shuffle(index)
        new_adj = new_adj[index]
        adj = adj[index]
        node = node[index]
        spatial = spatial[index]
        rel=rel[index]
        factor=factor[index]

        print (node.shape, adj.shape, new_adj.shape)

        return  node, spatial, adj, rel, factor, np.array(new_adj)


    return  node, spatial, np.array(new_adj), rel, factor


def cal_rel_dist(coords):
    rel = np.ones(shape=(coords.shape[0],coords.shape[1],coords.shape[1]),dtype=float)
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            for k in range(coords.shape[1]):
                rel[i][j][k] = ((coords[i][j][0]-coords[i][k][0])**2+(coords[i][j][1]-coords[i][k][1])**2+(coords[i][j][2]-coords[i][k][2])**2)**.5
    return rel

def load_data_protein(type_, path):
    if type_ == 'train':
        new_adj=np.load(path+'/edge_train.npy')
        spatial=np.load(path+'/node_train.npy')
        node=np.ones(shape=(spatial.shape[0],spatial.shape[1]),dtype=int)
        rel=cal_rel_dist(spatial)
        factor=np.array(range(1,1001)).reshape((1,1000))
        factor=np.tile(factor,38).reshape(-1)
        new_new_adj = []
        for adj in new_adj:
            x, y = np.where(adj)
            # print (x.shape, y.shape)
            edges = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
            new_new_adj_sub = []
            raw_edges = copy.deepcopy(edges)
            for i in range(FLAGS.sampling_num):
                edges = build_spanning_tree_edge(raw_edges, 'scipy', node.shape[1], len(raw_edges))
                adj = np.zeros_like(adj)
                adj[edges[0],edges[1]] = 1
                new_new_adj_sub.append(adj)
            new_new_adj.append(new_new_adj_sub)
        adj = np.array(new_new_adj)
        index = [i for i in range(len(node))]   #randomly shuffle the dataset
        np.random.shuffle(index)
        adj = adj[index]
        node = node[index]
        spatial = spatial[index]
        rel=rel[index]
        factor=factor[index]


    if type_ in ['test_generation','test_disentangle','test_reconstruct']:
        new_adj=np.load(path+'/edge_test.npy')
        spatial=np.load(path+'/node_test.npy')
        node=np.ones(shape=(spatial.shape[0],spatial.shape[1]),dtype=int)
        rel=cal_rel_dist(spatial)
        factor=np.array(range(1,1001)).reshape((1,1000))
        factor=np.tile(factor,38).reshape(-1)
        new_new_adj = []
        for adj in new_adj:
            x, y = np.where(adj)
            # print (x.shape, y.shape)
            edges = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
            new_new_adj_sub = []
            raw_edges = copy.deepcopy(edges)
            for i in range(FLAGS.sampling_num):
                edges = build_spanning_tree_edge(raw_edges, 'scipy', node.shape[1], len(raw_edges))
                adj = np.zeros_like(adj)
                adj[edges[0],edges[1]] = 1
                new_new_adj_sub.append(adj)
            new_new_adj.append(new_new_adj_sub)
        adj = np.array(new_new_adj)
        index = [i for i in range(len(node))]   #randomly shuffle the dataset
        np.random.shuffle(index)
        adj = adj[index]
        node = node[index]
        spatial = spatial[index]
        rel=rel[index]
        factor=factor[index]

    # new_adj=[]
    # for n in range(len(adj)):
    #     new_adj.append(adj[n])
    #     for i in range(len(new_adj[n])):
    #         new_adj[n][i,i]=0
    #         for j in range(len(new_adj[n])):
    #             assert new_adj[n][i,j]==new_adj[n][j,i] #check whether undirected graph


    return  node, spatial, adj, rel, factor, new_adj

def load_data_mnist(type_, path):
    #np_load_old = np.load
    #np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    # # of points to sample
    num_points = 50
    if type_ == 'train':
        f = open(path+'/mnist-combined-train-tasp_meshes.pickle','rb')
        data = pickle.load(f)
        adj,spatial=[],[]
        for i in range(5000): #len(data.data)
            spatial0 = data.data[i].sample_points(npoints=num_points)
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(np.array(spatial0))
            adj0=np.zeros(shape=(spatial0.shape[0],spatial0.shape[0]),dtype=int)
            mesh = pcd.compute_convex_hull()
            triangles = np.asarray(mesh[0].triangles)
            for j in range(triangles.shape[0]):
                adj0[triangles[j][0]][triangles[j][1]] = 1
                adj0[triangles[j][1]][triangles[j][2]] = 1
                adj0[triangles[j][0]][triangles[j][2]] = 1
                adj0[triangles[j][1]][triangles[j][0]] = 1
                adj0[triangles[j][2]][triangles[j][1]] = 1
                adj0[triangles[j][2]][triangles[j][0]] = 1                 
            spatial.append(spatial0)
            adj.append(adj0)
        spatial=np.array(spatial)
        adj=np.array(adj)
        node=np.ones(shape=(spatial.shape[0],spatial.shape[1]),dtype=int)
        rel=cal_rel_dist(spatial)
        index = [i for i in range(len(node))]   #randomly shuffle the dataset
        np.random.shuffle(index)
        adj = adj[index]
        node = node[index]
        spatial = spatial[index]
        rel=rel[index]
    
    if type_ in ['test_generation','test_disentangle','test_reconstruct']:
        f = open(path+'/mnist-combined-test-tasp_meshes.pickle','rb')
        data = pickle.load(f)
        adj,spatial=[],[]
        for i in range(1000): #len(data.data)
            spatial0 = data.data[i].sample_points(npoints=num_points)
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(np.array(spatial0))
            adj0=np.zeros(shape=(spatial0.shape[0],spatial0.shape[0]),dtype=int)
            mesh = pcd.compute_convex_hull()
            triangles = np.asarray(mesh[0].triangles)
            for j in range(triangles.shape[0]):
                adj0[triangles[j][0]][triangles[j][1]] = 1
                adj0[triangles[j][1]][triangles[j][2]] = 1
                adj0[triangles[j][0]][triangles[j][2]] = 1
                adj0[triangles[j][1]][triangles[j][0]] = 1
                adj0[triangles[j][2]][triangles[j][1]] = 1
                adj0[triangles[j][2]][triangles[j][0]] = 1                
            spatial.append(spatial0)
            adj.append(adj0)
        spatial=np.array(spatial)
        adj=np.array(adj)
        node=np.ones(shape=(spatial.shape[0],spatial.shape[1]),dtype=int)
        rel=cal_rel_dist(spatial)
        index = [i for i in range(len(node))]   #randomly shuffle the dataset
        np.random.shuffle(index)
        adj = adj[index]
        node = node[index]
        spatial = spatial[index]
        rel=rel[index]

    new_adj=[]
    for n in range(len(adj)):
        new_adj.append(adj[n])
        for i in range(len(new_adj[n])):
            new_adj[n][i,i]=0
            for j in range(len(new_adj[n])):
                assert new_adj[n][i,j]==new_adj[n][j,i] #check whether undirected graph


    return  node, spatial+np.ones((spatial.shape))*10, np.array(new_adj,dtype=int), rel



def to_one_hot(data, num_classes):
    one_hot = np.zeros(list(data.shape) + [num_classes])
    one_hot[np.arange(len(data)),data] = 1
    return one_hot

def load_data_scene(type_, path):
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    size = 10
    spatial, node, adj = [], [], []
    #node_feature = ['color', 'size', 'shape', 'material']
    node_feature = ['shape']
    rel_feature = ['right', 'behind', 'front', 'left']
    rel_feature_dou = [['12','21'], ['13','31'], ['24','42'], ['34','43']]
    color_feature = ['blue', 'gray', 'brown', 'purple', 'yellow', 'green', 'cyan', 'red']
    size_feature = ['large', 'small']
    shape_feature = ['sphere', 'cylinder', 'cube']
    material_feature = ['rubber', 'metal']
    features = [shape_feature]
    #features = [color_feature,size_feature,shape_feature,material_feature]
    #size_feature = [8, 2, 3, 2]
    size_feature = [3]
    if type_ == 'train':
        f = open(path+'/CLEVR_train_scenes.json', 'r')
        data = json.load(f)
        length = len(data['scenes'])
        for i in range(length):
            len_obj = len(data['scenes'][i]['objects'])
            if len_obj != size: continue
            spatial_sub = []
            node_sub = []
            for j in range(len_obj):
                coord = data['scenes'][i]['objects'][j]['3d_coords']
                spatial_sub.append(coord)
                node_sub_sub = np.array([[]])
                for feature in node_feature:
                    node_sub_sub = np.concatenate((node_sub_sub,to_one_hot(np.array([features[node_feature.index(feature)].index(data['scenes'][i]['objects'][j][feature])]), size_feature[node_feature.index(feature)])), axis=1)
                node_sub.append(node_sub_sub)
            node.append(node_sub)
            spatial.append(spatial_sub)
            adj_sub = np.zeros(shape=(size,size),dtype=int)
            relationship = data['scenes'][i]['relationships']
            merge_adj_sub = np.empty(shape=(size,size),dtype=object)
            merge_adj_sub[:,:] = ''
            for direction in relationship:
                for k in range(len(relationship[direction])):
                    for kl in range(len(relationship[direction][k])):
                        # adj_sub i, j means i is of *feature* j
                        merge_adj_sub[relationship[direction][k][kl]][k] += str(rel_feature.index(direction)+1)
                        adj_sub[relationship[direction][k][kl]][k] = rel_feature.index(direction)+1
            for direction in relationship:
                for k in range(len(relationship[direction])):
                    for kl in range(len(relationship[direction][k])):
                        # adj_sub i, j means i is of *feature* j
                        for ls in rel_feature_dou:
                            if merge_adj_sub[relationship[direction][k][kl]][k] in ls:
                                adj_sub[relationship[direction][k][kl]][k] = rel_feature_dou.index(ls)+1
            #print (adj_sub)
            #exit(0)
            adj.append(adj_sub)
    
        adj=np.array(adj)
        spatial=np.array(spatial)
        node=np.array(node)
        rel=cal_rel_dist(spatial)
        index = [i for i in range(len(node))]   #randomly shuffle the dataset
        np.random.shuffle(index)
        adj = adj[index]
        node = node[index]
        spatial = spatial[index]
        rel=rel[index]


    if type_ in ['test_generation','test_disentangle','test_reconstruct']:
        f = open(path+'/CLEVR_val_scenes.json', 'r')
        data = json.load(f)
        length = len(data['scenes'])
        for i in range(length):
            len_obj = len(data['scenes'][i]['objects'])
            if len_obj != size: continue
            spatial_sub = []
            node_sub = []
            for j in range(len_obj):
                coord = data['scenes'][i]['objects'][j]['3d_coords']
                spatial_sub.append(coord)
                node_sub_sub = np.array([[]])
                for feature in node_feature:
                    node_sub_sub = np.concatenate((node_sub_sub,to_one_hot(np.array([features[node_feature.index(feature)].index(data['scenes'][i]['objects'][j][feature])]), size_feature[node_feature.index(feature)])), axis=1)
                node_sub.append(node_sub_sub)
            node.append(node_sub)
            spatial.append(spatial_sub)
            adj_sub = np.zeros(shape=(size,size),dtype=int)
            relationship = data['scenes'][i]['relationships']
            for direction in relationship:
                for k in range(len(relationship[direction])):
                    for kl in range(len(relationship[direction][k])):
                        # adj_sub i, j means i is of *feature* j
                        adj_sub[relationship[direction][k][kl]][k] = rel_feature.index(direction)+1
            adj.append(adj_sub)

        adj=np.array(adj)
        spatial=np.array(spatial)
        node=np.array(node)
        rel=cal_rel_dist(spatial)
        index = [i for i in range(len(node))]   #randomly shuffle the dataset
        np.random.shuffle(index)
        adj = adj[index]
        node = node[index]
        spatial = spatial[index]
        rel=rel[index]
        sptial+np.ones((spatial.shape))*10
    return  node.reshape(-1,10,3), spatial, adj, rel
