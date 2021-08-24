from __future__ import division
from __future__ import print_function

import time
import os
import sys

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from sklearn import manifold
from scipy.special import expit

from optimizer import OptimizerVAE
from input_data import *

from preprocessing import *
from utils.visualizer import visualize_reconstruct, visualize_traverse, find_latent
from utils.utils import LossesLogger
from collections import defaultdict
from utils.evaluation import generation_evaluation, disentangle_evaluation,  reconstruct_evaluation



def sigmoid(x):
        return 1 / (1 + np.exp(-x))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True

flags = tf.app.flags
FLAGS = flags.FLAGS
 
flags.DEFINE_integer('spatial_conv_layers', 3, 'Number of spatial_conv_layers.')
flags.DEFINE_list('s_channel', [10,10,20], 'Number of channles in spatial convolution.')
flags.DEFINE_list('s_kernel_size', [5,5,5], 'size of kernel in each spatial conv layer.')
flags.DEFINE_list('s_strides', [1,1,1], 'Number of strides in  each spatial conv layer.')
flags.DEFINE_integer('s_hidden_size', 100, 'length of the hidden layer of spatial.')
flags.DEFINE_integer('s_latent_size', 100, 'length of the latent representation of spatial.')
#graph encoder
flags.DEFINE_integer('graph_conv_layers', 2, 'Number of graph_conv_layers.')
flags.DEFINE_list('g_conv_hidden', [10,20], 'Number of strides in  each spatial conv layer.')
flags.DEFINE_integer('g_hidden_size', 100, 'length of the hidden layer of spatial.')
flags.DEFINE_integer('g_latent_size', 100, 'length of the latent representation of graph.')
#spatial graph encoder
flags.DEFINE_integer('spatial_graph_conv_layers', 2, 'Number of spatial-graph_conv_layers.')
flags.DEFINE_list('sg_conv_hidden', [[20,20,20],[50,50,50]], 'length of hidden size in each spatial-graph conv layer.')
flags.DEFINE_integer('sg_hidden_size', 200, 'length of the hidden layer of spatial-graph.')
flags.DEFINE_integer('sg_latent_size', 200, 'length of the latent representation of sptial graph.')
#spatial decoder
flags.DEFINE_integer('spatial_deconv_layers', 3, 'Number of spatial_deconv_layers.')
flags.DEFINE_list('s_d_channel', [50,20,10], 'Number of channles in spatial deconvolution.')
flags.DEFINE_list('s_d_kernel_size', [5,5,5], 'size of kernel in each spatial deconv layer.')
flags.DEFINE_list('s_d_strides', [1,1,1], 'Number of strides in  each spatial deconv layer.')
#graph decoder
flags.DEFINE_integer('graph_deconv_layers', 2, 'Number of graph_deconv_layers.')
flags.DEFINE_list('n_d_channel', [50,20,10], 'Number of channles in node deconvolution.')
flags.DEFINE_list('n_d_kernel_size', [5,5,5], 'size of kernel in each node deconv layer.')
flags.DEFINE_list('n_d_strides', [1,1,1], 'Number of strides in  each node deconv layer.')
flags.DEFINE_integer('d_hidden_size', 20, 'length of the hidden layer of graph deconv.')
flags.DEFINE_list('e_d_hidden', [50,20,10], 'Number of channles in sedge deconvolution.')

flags.DEFINE_integer('node_h_size', 20, 'node latent size in decoder part.')
flags.DEFINE_string('model_type', 'disentangled', 'base, disentangled, disentangled_C,NED-VAE-IP,beta-TCVAE, geoGCN, posGCN')

#training paramters
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 1, 'keep probability.')
flags.DEFINE_integer('batch_size', 2, 'Number of samples in a batch.')
flags.DEFINE_integer('decoder_batch_size',2, 'Number of samples in a batch.')
flags.DEFINE_integer('sg_batch_size', 5, 'Number of samples in a batch.')
flags.DEFINE_integer('sg_decoder_batch_size',5, 'Number of samples in a batch.')
flags.DEFINE_string('dataset_path', '../dataset/', 'Number of samples in a batch.')
flags.DEFINE_integer('num_feature', 1, 'Number of features.')
flags.DEFINE_integer('spatial_dim', 2, 'The dimension of spatial information.')
flags.DEFINE_integer('verbose', 1, 'Output all epoch data')
flags.DEFINE_integer('test_count', 10, 'batch of tests')
flags.DEFINE_string('model', 'feedback', 'Model string.')
flags.DEFINE_integer('seeded', 1, 'Set numpy random seed')
flags.DEFINE_integer('connected_split', 0, 'use split with training set always connected')
flags.DEFINE_string('type', 'test_reconstruct', 'train or test')
flags.DEFINE_integer('if_traverse', 1, 'varying the z to see the generated graphs')
flags.DEFINE_integer('visualize_length', 5, 'varying the z to see the generated graphs')
flags.DEFINE_string('dataset', 'synthetic2', 'synthetic1 or synthetic2')

flags.DEFINE_float('C_max', 100, 'capacity parameter(C) of bottleneck channel')
flags.DEFINE_float('C_stop_iter', 1e2, 'when to stop increasing the capacity')
flags.DEFINE_float('gamma', 100, 'gamma parameter for KL-term in understanding beta-VAE')
flags.DEFINE_float('C_step', 20, 'every c_step epoch the C changes')

flags.DEFINE_integer('sampling_num', 10, 'sampling ten times')

flags.DEFINE_integer('dim', None, 'dim for traverse')
flags.DEFINE_string('group_type', None, 'group type for traverse')

if FLAGS.model_type=='base':
   from model_joint import *
else:
   from model import *

def ZscoreNormalization(x, mean_, std_):
    """Z-score normaliaztion"""
    x = (x - mean_) / std_
    return x


def main(beta, type_model):
        if 'vae_type' in list(flags.FLAGS):
            delattr(flags.FLAGS,'vae_type')
        flags.DEFINE_string('vae_type', type_model, 'local or global or local_global')

        if FLAGS.type =='test_disentangle':
            FLAGS.batch_size=FLAGS.visualize_length*(FLAGS.s_latent_size+FLAGS.g_latent_size+FLAGS.sg_latent_size)
            FLAGS.decoder_batch_size=FLAGS.batch_size
        if FLAGS.seeded:
            np.random.seed(1)

        # Load data
        if FLAGS.dataset == 'synthetic1':
           dataset_path=FLAGS.dataset_path+'spatial_network_correlated1/25'
           if True:
            feature,spatial, adj, rel, factor, adj_truth = load_data_syn(FLAGS.type, dataset_path)
            adj = adj.reshape(-1,adj.shape[-2],adj.shape[-1])
            print (adj.shape)
           else:
            feature,spatial, adj, rel, factor= load_data_syn(FLAGS.type, dataset_path)
           FLAGS.spatial_conv_layers=3
           FLAGS.s_channel=[10,10,20]
           FLAGS.s_kernel_size=[5,5,5]
           FLAGS.s_strides=[1,1,1]
           FLAGS.s_hidden_size=100
           FLAGS.s_latent_size=100
           #graph encoder
           FLAGS.graph_conv_layers= 2
           FLAGS.g_conv_hidden=[10,20]
           FLAGS.g_hidden_size=100
           FLAGS.g_latent_size=100
           #spatial graph encoder
           FLAGS.spatial_graph_conv_layers=2
           FLAGS.sg_conv_hidden=[[20,20,20],[50,50,50]]
           FLAGS.sg_hidden_size=500
           FLAGS.sg_latent_size=500
           #spatial decoder
           FLAGS.spatial_deconv_layers=3
           FLAGS.s_d_channel=[50,20,10]
           FLAGS.s_d_kernel_size=[5,5,5]
           FLAGS.s_d_strides=[1,1,1]
           #graph decoder
           FLAGS.graph_deconv_layers=2
           FLAGS.n_d_channel=[50,20,10]
           FLAGS.n_d_kernel_size=[5,5,5]
           FLAGS.n_d_strides=[1,1,1]
           FLAGS.d_hidden_size=20
           FLAGS.e_d_hidden=[50,20,10]
           FLAGS.node_h_size=50
           #training paramters
           FLAGS.learning_rate=0.001
           FLAGS.epochs=1000
           FLAGS.dropout=1
           FLAGS.batch_size=10
           FLAGS.decoder_batch_size=10
           FLAGS.sg_batch_size=10
           FLAGS.sg_decoder_batch_size=10
        elif FLAGS.dataset == 'synthetic2':
           dataset_path=FLAGS.dataset_path+'spatial_network_correlated2/25'
           if True:
            feature,spatial, adj, rel, factor, adj_truth = load_data_syn(FLAGS.type, dataset_path)
            adj = adj.reshape(-1,adj.shape[-2],adj.shape[-1])
            print (adj.shape)
           else:
            feature,spatial, adj, rel, factor= load_data_syn(FLAGS.type, dataset_path)
           FLAGS.spatial_conv_layers=3
           FLAGS.s_channel=[10,10,20]
           FLAGS.s_kernel_size=[5,5,5]
           FLAGS.s_strides=[1,1,1]
           FLAGS.s_hidden_size=100
           FLAGS.s_latent_size=100
           #graph encoder
           FLAGS.graph_conv_layers= 2
           FLAGS.g_conv_hidden=[10,20]
           FLAGS.g_hidden_size=100
           FLAGS.g_latent_size=100
           #spatial graph encoder
           FLAGS.spatial_graph_conv_layers=2
           FLAGS.sg_conv_hidden=[[20,20,20],[50,50,50]]
           FLAGS.sg_hidden_size=100
           FLAGS.sg_latent_size=100
           #spatial decoder
           FLAGS.spatial_deconv_layers=3
           FLAGS.s_d_channel=[50,20,10]
           FLAGS.s_d_kernel_size=[5,5,5]
           FLAGS.s_d_strides=[1,1,1]
           #graph decoder
           FLAGS.graph_deconv_layers=2
           FLAGS.n_d_channel=[50,20,10]
           FLAGS.n_d_kernel_size=[5,5,5]
           FLAGS.n_d_strides=[1,1,1]
           FLAGS.d_hidden_size=20
           FLAGS.e_d_hidden=[50,20,10]
           FLAGS.node_h_size=20
           #training paramters
           FLAGS.learning_rate=0.0008
           FLAGS.epochs=1000
           FLAGS.dropout=1
           FLAGS.batch_size=10
           FLAGS.decoder_batch_size=10
           FLAGS.sg_batch_size=10
           FLAGS.sg_decoder_batch_size=10
        elif FLAGS.dataset == 'protein':
           FLAGS.spatial_dim = 3
           dataset_path=FLAGS.dataset_path+'protein'
           feature,spatial, adj, rel, factor, adj_truth = load_data_protein(FLAGS.type, dataset_path)
           adj = adj.reshape(-1,adj.shape[-2],adj.shape[-1])
           FLAGS.sg_conv_hidden = [[10,10,10,10],[20,20,20,20]]
           FLAGS.sg_hidden_size=50
           FLAGS.sg_latent_size=50
           FLAGS.s_hidden_size=5
           FLAGS.s_latent_size=5   
           FLAGS.g_hidden_size=5
           FLAGS.g_latent_size=5            
           FLAGS.node_h_size=5 
           FLAGS.s_channel = [10,10,20]
           FLAGS.s_kernel_size=[5,5,5]
           FLAGS.batch_size=50
           FLAGS.decoder_batch_size=50
           FLAGS.sg_batch_size=50
           FLAGS.sg_decoder_batch_size=50
        elif FLAGS.dataset == 'mnist':
            FLAGS.spatial_dim = 3
            dataset_path=FLAGS.dataset_path+'3D_mesh'
            feature,spatial, adj, rel = load_data_mnist(FLAGS.type, dataset_path)
            FLAGS.sg_conv_hidden = [[20,20,20,20],[50,50,50,50]]

        num_nodes = adj.shape[1]

        num_features = FLAGS.num_feature
        pos_weight = float(adj.shape[0] *adj.shape[1] * adj.shape[1] - adj.sum()) / adj.sum()
        norm = adj.shape[0] *adj.shape[1] * adj.shape[1] / float((adj.shape[0] *adj.shape[1] * adj.shape[1] - adj.sum()) * 2)

        feature=feature.reshape([-1,num_nodes,num_features])
        rel=rel.reshape([-1,num_nodes,num_nodes,1])

        if True:
          placeholders = {
                'features': tf.placeholder(tf.float32,[FLAGS.batch_size*FLAGS.sampling_num,num_nodes,num_features]),
                'spatial': tf.placeholder(tf.float32,[FLAGS.batch_size*FLAGS.sampling_num,num_nodes,FLAGS.spatial_dim]),
                'adj': tf.placeholder(tf.float32,[FLAGS.batch_size*FLAGS.sampling_num,adj.shape[1],adj.shape[2]]),
                'adj_truth': tf.placeholder(tf.float32,[FLAGS.batch_size,adj.shape[1],adj.shape[2]]),
                'feature_truth': tf.placeholder(tf.float32,[FLAGS.batch_size,num_nodes,num_features]),
                'spatial_truth': tf.placeholder(tf.float32,[FLAGS.batch_size,num_nodes,FLAGS.spatial_dim]),
                'rel_truth':  tf.placeholder(tf.float32,[FLAGS.batch_size,adj.shape[1],adj.shape[2], 1]),
                'rel':  tf.placeholder(tf.float32,[FLAGS.batch_size*FLAGS.sampling_num,adj.shape[1],adj.shape[2], 1]),
                'dropout': tf.placeholder_with_default(0., shape=()),
                'global_iter': tf.placeholder_with_default(0., shape=()),
          }
        else:
          placeholders = {
                  'features': tf.placeholder(tf.float32,[FLAGS.batch_size,num_nodes,num_features]),
                  'spatial': tf.placeholder(tf.float32,[FLAGS.batch_size,num_nodes,FLAGS.spatial_dim]),
                  'adj': tf.placeholder(tf.float32,[FLAGS.batch_size,adj.shape[1],adj.shape[2]]),
                  'rel':  tf.placeholder(tf.float32,[FLAGS.batch_size,adj.shape[1],adj.shape[2], 1]),
                  'dropout': tf.placeholder_with_default(0., shape=()),
                  'global_iter': tf.placeholder_with_default(0., shape=()),
            }

        if FLAGS.type != 'test_disentangle':
            model = SGCNModelVAE(placeholders, num_features, num_nodes)

        TRAIN_LOSSES_LOGFILE='./train_loss_'+FLAGS.dataset+'_'+FLAGS.model_type+'.txt'

        losses_logger = LossesLogger(os.path.join(TRAIN_LOSSES_LOGFILE))


        if FLAGS.type=='train':
          with tf.name_scope('optimizer'):
                opt = OptimizerVAE(preds_edge=model.generated_adj_prob,
                                   preds_node=model.generated_node_feat,
                                   preds_spatial=model.generated_spatial,
                                   labels_edge=placeholders['adj_truth'],
                                   labels_node=placeholders['feature_truth'],
                                   labels_spatial=placeholders['spatial_truth'],
                                   labels_rel=placeholders['rel_truth'],
                                   global_iter=placeholders['global_iter'],
                                   model=model, num_nodes=num_nodes,
                                   pos_weight=pos_weight,
                                   norm=norm,
                                   beta=beta)

        if FLAGS.type != 'test_disentangle':
            saver = tf.train.Saver()
        if FLAGS.type=='train':
          with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
             # Train model
            feature_truth = feature 
            spatial_truth = spatial 
            rel_truth = rel
            feature = np.tile(feature, (FLAGS.sampling_num,1,1))
            spatial = np.tile(spatial, (FLAGS.sampling_num,1,1))
            rel = np.tile(rel, (FLAGS.sampling_num,1,1,1))
            for epoch in range(FLAGS.epochs):
              storer = defaultdict(list)
              batch_num=int(adj.shape[0]/(FLAGS.batch_size*FLAGS.sampling_num))
              check=[]
              epoch_time = time.time()
              for i in range(batch_num):
                  adj_batch=adj[i*FLAGS.batch_size*FLAGS.sampling_num:i*FLAGS.batch_size*FLAGS.sampling_num+FLAGS.batch_size*FLAGS.sampling_num]
                  feature_batch=feature[i*FLAGS.batch_size*FLAGS.sampling_num:i*FLAGS.batch_size*FLAGS.sampling_num+FLAGS.batch_size*FLAGS.sampling_num]
                  spatial_batch=spatial[i*FLAGS.batch_size*FLAGS.sampling_num:i*FLAGS.batch_size*FLAGS.sampling_num+FLAGS.batch_size*FLAGS.sampling_num]
                  rel_batch=rel[i*FLAGS.batch_size*FLAGS.sampling_num:i*FLAGS.batch_size*FLAGS.sampling_num+FLAGS.batch_size*FLAGS.sampling_num]
                  adj_truth_batch=adj_truth[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                  feature_truth_batch=feature_truth[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                  spatial_truth_batch=spatial_truth[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                  rel_truth_batch=rel_truth[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]

                  t = time.time()
                  # Construct feed dictionary
                  feed_dict = construct_feed_dict_train(feature_batch, spatial_batch, adj_batch, rel_batch, adj_truth_batch, feature_truth_batch, spatial_truth_batch, rel_truth_batch, placeholders)
                  feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                  feed_dict.update({placeholders['global_iter']: epoch})
                  # Run single weight update
                  outs = sess.run([opt.opt_op, opt.overall_loss, model.generated_adj], feed_dict=feed_dict)
                  # Compute average loss
                  overall_loss=outs[1]
                  acc=sum(sum(sum(outs[2]==adj_truth_batch)))/(FLAGS.batch_size*num_nodes*num_nodes)
                  storer['loss'].append(overall_loss[0])
                  storer['spatial_loss'].append(overall_loss[1])
                  storer['adj_loss'].append(overall_loss[2])
                  storer['adj_acc'].append(acc)
                  storer['node_loss'].append(overall_loss[3])
                  if FLAGS.model_type in ['disentangled','disentangled_C','NED-VAE-IP','beta-TCVAE']:
                      storer['graph_kl'].append(overall_loss[4])
                      storer['spatial_kl'].append(overall_loss[5])
                      storer['sg_kl'].append(overall_loss[6])
                  else:
                      storer['sg_kl'].append(overall_loss[4])
                  check.append(outs[2])

                  print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.5f}".format(overall_loss[0]),
                  "time=", "{:.5f}".format(time.time() - t))
              print ("epoch time=", "{:.5f}".format(time.time()-epoch_time))
              if epoch%100==0:
                save_path = saver.save(sess, "/home/ydu6/generation_eff_latent_sg/src/tmp/"+FLAGS.dataset+'_'+FLAGS.model_type+"/model_dgt_global_"+str(epoch)+".ckpt")
              losses_logger.log(epoch, storer)

            print("Optimization Finished!")
            return np.array(check), adj

        def generate_new_train(feed_dic):
           feed_dict = feed_dic
           feed_dict.update({placeholders['dropout']: 1.0})
           z_s,z_sg,z_g, adj, spatial, node = sess.run([model.z_mean_s,model.z_mean_sg,model.z_mean_g, model.generated_adj, model.generated_spatial, model.generated_node_feat], feed_dict=feed_dict)
           return z_s,z_sg,z_g, adj, spatial, node

        def generate_new(feature_batch, spatial_batch, adj_batch,rel_batch):
           feed_dict = construct_feed_dict(feature_batch, spatial_batch, adj_batch, rel_batch, placeholders)
           feed_dict.update({placeholders['dropout']: 1.0})
           if FLAGS.model_type == "base":
             z_sg, adj, node, spatial = sess.run([model.z_mean_sg, model.generated_adj,model.generated_node_feat,model.generated_spatial], feed_dict=feed_dict)
             return z_sg, adj, spatial, node
           else:
             z_s,z_sg,z_g, adj, spatial, node = sess.run([model.z_mean_s,model.z_mean_sg,model.z_mean_g, model.generated_adj, model.generated_spatial, model.generated_node_feat], feed_dict=feed_dict)
             return z_s,z_sg,z_g, adj, spatial, node

        if FLAGS.type =='test_reconstruct':
          with tf.Session() as sess:
            saver.restore(sess, "/home/ydu6/generation_eff_latent_sg/src/tmp/"+FLAGS.dataset+'_'+FLAGS.model_type+"/model_dgt_global_"+str(480)+".ckpt")
            print("Model restored.")
            generated_adj=[]
            generated_nodes=[]
            generated_spatial=[]
            batch_num=int(adj.shape[0]/(FLAGS.batch_size*FLAGS.sampling_num))
            feature_truth = feature 
            spatial_truth = spatial 
            rel_truth = rel
            feature = np.tile(feature, (FLAGS.sampling_num,1,1))
            spatial = np.tile(spatial, (FLAGS.sampling_num,1,1))
            rel = np.tile(rel, (FLAGS.sampling_num,1,1,1))
            #generated_adj_prob=[]
            z_s=[]
            z_sg=[]
            z_g=[]
            for i in range(batch_num):
                adj_batch=adj[i*FLAGS.batch_size*FLAGS.sampling_num:i*FLAGS.batch_size*FLAGS.sampling_num+FLAGS.batch_size*FLAGS.sampling_num]
                feature_batch=feature[i*FLAGS.batch_size*FLAGS.sampling_num:i*FLAGS.batch_size*FLAGS.sampling_num+FLAGS.batch_size*FLAGS.sampling_num]
                spatial_batch=spatial[i*FLAGS.batch_size*FLAGS.sampling_num:i*FLAGS.batch_size*FLAGS.sampling_num+FLAGS.batch_size*FLAGS.sampling_num]
                rel_batch=rel[i*FLAGS.batch_size*FLAGS.sampling_num:i*FLAGS.batch_size*FLAGS.sampling_num+FLAGS.batch_size*FLAGS.sampling_num]
                adj_truth_batch=adj_truth[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                feature_truth_batch=feature_truth[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                spatial_truth_batch=spatial_truth[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                rel_truth_batch=rel_truth[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                feed_dict = construct_feed_dict_train(feature_batch, spatial_batch, adj_batch, rel_batch, adj_truth_batch, feature_truth_batch, spatial_truth_batch, rel_truth_batch, placeholders)
                z_s_batch,z_sg_batch, z_g_batch, generated_adj_, generated_spatial_, generated_node_= generate_new_train(feed_dict)
                generated_adj.append(generated_adj_)
                generated_nodes.append(generated_node_)
                generated_spatial.append(generated_spatial_)
                z_s.append(z_s_batch.reshape((FLAGS.batch_size,-1)))
                z_sg.append(z_sg_batch.reshape((FLAGS.batch_size,FLAGS.sampling_num,-1)).mean(axis=1))
                z_g.append(z_g_batch.reshape((FLAGS.batch_size,-1)))


            if FLAGS.model_type == "base":
              np.save('./qualitative_evaluation/'+str(FLAGS.dataset)+"/"+FLAGS.vae_type+'_z_sg.npy',np.array(z_sg))
            else:
              np.save('./qualitative_evaluation/'+str(FLAGS.dataset)+"/"+FLAGS.vae_type+'_z_s.npy',np.array(z_s))
              np.save('./qualitative_evaluation/'+str(FLAGS.dataset)+"/"+FLAGS.vae_type+'_z_sg.npy',np.array(z_sg))
              np.save('./qualitative_evaluation/'+str(FLAGS.dataset)+"/"+FLAGS.vae_type+'_z_g.npy',np.array(z_g))
            print (len(z_s))
            print (z_s[0].shape)
            generated_adj=np.array(generated_adj).reshape(-1,num_nodes,num_nodes)
            generated_nodes=np.array(generated_nodes).reshape(-1,num_nodes,num_features)
            generated_spatial=np.array(generated_spatial).reshape(-1,num_nodes,FLAGS.spatial_dim)
            #visualize_reconstruct(5, adj_batch, feature_batch*120, spatial_batch*600, generated_adj, generated_nodes*120, generated_spatial*600)
            evaluate_results=reconstruct_evaluation(generated_adj,generated_nodes,generated_spatial, adj_truth, feature_truth, spatial_truth, FLAGS.dataset)
            disentangle_results=disentangle_evaluation(z_s, z_g, z_sg, factor, FLAGS.dataset)
            print (evaluate_results,disentangle_results)
            return  disentangle_results

        if FLAGS.type =='test_generation':
          with tf.Session() as sess:
            saver.restore(sess, "/home/ydu6/generation_eff_latent_sg/src/tmp/"+FLAGS.dataset+'_'+FLAGS.model_type+"/model_dgt_global_"+str(480)+".ckpt")
            print("Model restored.")
            generated_adj=[]
            generated_nodes=[]
            generated_spatial=[]
            batch_num=int(adj.shape[0]/(FLAGS.batch_size*FLAGS.sampling_num))
            feature_truth = feature 
            spatial_truth = spatial 
            rel_truth = rel
            feature = np.tile(feature, (FLAGS.sampling_num,1,1))
            spatial = np.tile(spatial, (FLAGS.sampling_num,1,1))
            rel = np.tile(rel, (FLAGS.sampling_num,1,1,1))
            z_s=[]
            z_sg=[]
            z_g=[]
            for i in range(batch_num):
                adj_batch=adj[i*FLAGS.batch_size*FLAGS.sampling_num:i*FLAGS.batch_size*FLAGS.sampling_num+FLAGS.batch_size*FLAGS.sampling_num]
                feature_batch=feature[i*FLAGS.batch_size*FLAGS.sampling_num:i*FLAGS.batch_size*FLAGS.sampling_num+FLAGS.batch_size*FLAGS.sampling_num]
                spatial_batch=spatial[i*FLAGS.batch_size*FLAGS.sampling_num:i*FLAGS.batch_size*FLAGS.sampling_num+FLAGS.batch_size*FLAGS.sampling_num]
                rel_batch=rel[i*FLAGS.batch_size*FLAGS.sampling_num:i*FLAGS.batch_size*FLAGS.sampling_num+FLAGS.batch_size*FLAGS.sampling_num]
                adj_truth_batch=adj_truth[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                feature_truth_batch=feature_truth[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                spatial_truth_batch=spatial_truth[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                rel_truth_batch=rel_truth[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                feed_dict = construct_feed_dict_train(feature_batch, spatial_batch, adj_batch, rel_batch, adj_truth_batch, feature_truth_batch, spatial_truth_batch, rel_truth_batch, placeholders)
                z_s_batch,z_sg_batch, z_g_batch, generated_adj_, generated_spatial_, generated_node_= generate_new_train(feed_dict)
                generated_adj.append(generated_adj_)
                generated_nodes.append(generated_node_)
                generated_spatial.append(generated_spatial_)
                z_s.append(z_s_batch.reshape((FLAGS.batch_size,-1)))
                z_sg.append(z_sg_batch.reshape((FLAGS.batch_size,FLAGS.sampling_num,-1)).mean(axis=1))
                z_g.append(z_g_batch.reshape((FLAGS.batch_size,-1)))

            generated_adj=np.array(generated_adj).reshape(-1,num_nodes,num_nodes)
            generated_nodes=np.array(generated_nodes).reshape(-1,num_nodes,num_features)
            generated_spatial=np.array(generated_spatial).reshape(-1,num_nodes,FLAGS.spatial_dim)
            #visualize_reconstruct(5, adj_batch, feature_batch*120, spatial_batch*600, generated_adj, generated_nodes*120, generated_spatial*600)
            evaluate_results=generation_evaluation(generated_adj,generated_nodes,generated_spatial, adj, feature, spatial, FLAGS.dataset)
            print (evaluate_results)
            return  evaluate_results



        if FLAGS.type== 'test_disentangle':
          with tf.Session() as sess:
            generated_adj=[]
            generated_nodes=[]
            generated_spatial=[]
            adj_batch=adj[:FLAGS.batch_size]
            feature_batch=feature[:FLAGS.batch_size]
            spatial_batch=spatial[:FLAGS.batch_size]
            rel_batch=rel[:FLAGS.batch_size]
            model = SGCNModelVAE(placeholders, num_features, num_nodes, group_type=FLAGS.group_type, dim=FLAGS.dim, dim_a=77, dim_b=48, dim_c=171)
            saver = tf.train.Saver()
            saver.restore(sess, "/home/ydu6/generation/src/tmp/"+FLAGS.dataset+'_'+FLAGS.model_type+"/model_dgt_global_"+str(300)+".ckpt")
            print("Model restored.")

            z_s,z_sg, z_g, generated_adj, generated_spatial, generated_nodes = generate_new(feature_batch, spatial_batch, adj_batch, rel_batch)
            generated_adj=np.array(generated_adj).reshape([-1, num_nodes, num_nodes])
            generated_nodes=np.array(generated_nodes).reshape([-1, num_nodes, num_features])
            generated_spatial=np.array(generated_spatial).reshape([-1, num_nodes,FLAGS.spatial_dim])
            print (generated_adj.shape, generated_nodes.shape, generated_spatial.shape)
            min_n, max_n = np.min(generated_nodes[FLAGS.visualize_length:FLAGS.visualize_length*2]*120), np.max(generated_nodes[FLAGS.visualize_length:FLAGS.visualize_length*2]*120)
            generated_nodes[FLAGS.visualize_length:FLAGS.visualize_length*2] = (generated_nodes[FLAGS.visualize_length:FLAGS.visualize_length*2]*120-min_n)/(max_n-min_n)
            # print (np.min(generated_nodes[FLAGS.visualize_length:FLAGS.visualize_length*2]),np.max(generated_nodes[FLAGS.visualize_length:FLAGS.visualize_length*2]))
            print (np.min(generated_spatial[:FLAGS.visualize_length]*600), np.max(generated_spatial[:FLAGS.visualize_length]*600))
            print (np.min(generated_spatial)*600, np.max(generated_spatial)*600)
            visualize_traverse(generated_adj, generated_nodes*120, generated_spatial*600,1,FLAGS.visualize_length,FLAGS.dataset)
            # print('spatial:'+str(a))
            # print('graph:'+str(b))
            # print('joint:'+str(c))

if __name__ == '__main__':
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    models=['disentangled']   #'posGCN','geoGCN','disentangled_C''NED-VAE-IP','beta-TCVAE','disentangled','base',,,,,,'InfoVAE',,,   ,'HFVAE'],,,,'InfoVAE''FactorVAE''InfoVAE','DIP-VAE''FactorVAE','HFVAE'
    types= ['train','test_reconstruct','test_generation']
    generation_results={}
    reconstruct_results={}
    for type_ in types:
        FLAGS.type=type_
        for t in models:
          tf.reset_default_graph()
          FLAGS.model_type=t
          if FLAGS.type =='train':
             main(1,t)
          elif FLAGS.type =='test_reconstruct':
             reconstruct_result=main(1,t)
             reconstruct_results[t]=reconstruct_result
          elif FLAGS.type =='test_generation':
             generation_result=main(1,t)
             generation_results[t]=generation_result
          else:
             main(1,t)
    print(generation_results)
    print(reconstruct_results)
