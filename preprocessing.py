import numpy as np
import scipy.sparse as sp
import networkx as nx
import itertools


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph_coo(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    return sparse_to_tuple(adj_normalized)

def construct_feed_dict_train(features, spatial, adj, rel, adj_truth, feature_truth_batch, spatial_truth_batch, rel_truth_batch, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['spatial']: spatial})
    feed_dict.update({placeholders['rel']: rel})
    feed_dict.update({placeholders['adj_truth']: adj_truth})
    feed_dict.update({placeholders['feature_truth']: feature_truth_batch})
    feed_dict.update({placeholders['spatial_truth']: spatial_truth_batch})
    feed_dict.update({placeholders['rel_truth']: rel_truth_batch})
    return feed_dict

def construct_feed_dict(features, spatial, adj, rel, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['spatial']: spatial})
    feed_dict.update({placeholders['rel']: rel})
    return feed_dict

def pick_edges(graph, count):
    G = nx.Graph(graph)
    edges = []
    while len(edges) < count:
        G_edges = G.edges()
        i = np.random.randint(len(G_edges))
        u, v = G_edges[i]
        G.remove_edge(u, v)

        if nx.has_path(G, u, v):
            edges.append([min(u,v), max(u,v)])
        else:
            G.add_edge(u, v)
    return edges

def pick_false_edges(graph, count):
    G = nx.Graph(graph)
    edges = []
    while len(edges) < count:
        G_nodes = G.nodes()
        i = np.random.randint(len(G_nodes))
        j = np.random.randint(len(G_nodes))
        u = G_nodes[i]
        v = G_nodes[j]

        if v not in G.neighbors(u) + [u]:
            edges.append([min(u,v), max(u,v)])
            G.add_edge(u, v)
    return edges

def get_test_edges(adj):
    '''Randomly sample connected training graph, and validation and test edges.'''
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    edges_all = sparse_to_tuple(adj)[0].tolist()

    edge_count = len(edges_all) / 2.0
    num_test = int(np.floor(edge_count / 10.))
    num_val = int(np.floor(edge_count / 20.))

    G = nx.to_networkx_graph(adj)
    test_edges = pick_edges(G, num_test)
    test_edges_false = pick_false_edges(G, num_test)

    G.remove_edges_from(test_edges)
    val_edges = pick_edges(G, num_val)
    val_edges_false = pick_false_edges(G, num_val)

    G.remove_edges_from(val_edges)
    adj_train = nx.to_scipy_sparse_matrix(G)
    train_edges = sparse_to_tuple(adj_train)[0].tolist()

    def ismember(a, b):
        seta = set([tuple(x) for x in a])
        setb = set([tuple(x) for x in b])
        return len(seta & setb) > 0

    assert not ismember(test_edges_false, edges_all)
    assert not ismember(val_edges_false, val_edges + train_edges)
    assert not ismember(val_edges, train_edges)
    assert not ismember(test_edges, train_edges)
    assert not ismember(val_edges, test_edges)
    assert ismember(val_edges, val_edges)

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def edge_dropout(adj, dropout):

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    num_val = int(np.floor(edges.shape[0] * 1.0 * dropout))

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    train_edges = np.delete(edges, val_edge_idx, axis=0)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    return adj_train

