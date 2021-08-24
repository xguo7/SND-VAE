# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:02:31 2019

@author: gxjco

this code is node edge disentangled VAE model (basement)
"""

from layers import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS




class SGCNModelVAE(object):

    '''VGAE Model for reconstructing graph edges from node representations.'''
    def __init__(self, placeholders, num_features, num_nodes, group_type=None, dim=None, dim_a=None, dim_b=None, dim_c=None, **kwargs):
        super(SGCNModelVAE, self).__init__(**kwargs)
        self.node_feature = placeholders['feature_truth']  #node attributes B*N*h
        self.node_feature_sg = placeholders['features']
        self.inputs_3d = placeholders['spatial_truth']  #B*N*3
        self.input_dim = num_features
        self.n_samples = num_nodes
        self.adj = placeholders['adj_truth']   #B*N*N (element [i,j] refers to whether ane edge between i  and j )
        self.rel = placeholders['rel_truth']   #B*N*N (element [i,j] refers to spatial distance between i  and j )
        self.adj_sg = placeholders['adj']   
        self.rel_sg = placeholders['rel']
        self.dropout = placeholders['dropout']
        self.weight_norm = 0
        self.group_type = group_type
        self.dim = dim
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.dim_c = dim_c

        batch_norm=tf.keras.layers.BatchNormalization
        # graph batch norm layers
        self.g_bn_s=[]
        for i in range(FLAGS.spatial_conv_layers):
          self.g_bn_s.append(batch_norm(name='g_bn_s'+str(i)))

        self.g_bn_g=[]
        for i in range(FLAGS.graph_conv_layers):
          self.g_bn_g.append(batch_norm(name='g_bn_g'+str(i)))

        self.g_bn_sg=[]
        for i in range(FLAGS.spatial_graph_conv_layers):
          self.g_bn_sg.append(batch_norm(name='g_bn_sg'+str(i)))

        self.d_bn_e=[]
        for i in range(FLAGS.graph_deconv_layers):
          self.d_bn_e.append(batch_norm(name='d_bn_e'+str(i)))

        self.d_bn_s=[]
        for i in range(FLAGS.spatial_deconv_layers):
          self.d_bn_s.append(batch_norm(name='d_bn_s'+str(i)))

        self.d_bn_n=[]
        for i in range(FLAGS.spatial_deconv_layers):
          self.d_bn_n.append(batch_norm(name='d_bn_n'+str(i)))

        self.encoder_s = batch_norm(name='encoder_s')
        self.encoder_sg = batch_norm(name='encoder_sg')
        self.encoder_g = batch_norm(name='encoder_g')
        self.decoder_node = batch_norm(name='decoder_node')
        self.decoder_adj = batch_norm(name='decoder_adj')

        self._build()


    def _build(self):
        self.encoder()
        self.z_s,self.z_sg,self.z_g = self.get_z(random = True)
        if FLAGS.type=='train':
          self.generated_adj, self.generated_adj_prob, self.generated_spatial,  self.generated_node_feat = self.decoder(self.z_s,self.z_sg,self.z_g)
        if FLAGS.type=='test_reconstruct':
          self.generated_adj, self.generated_adj_prob, self.generated_spatial,  self.generated_node_feat= self.sample(self.z_s,self.z_sg,self.z_g)
        if FLAGS.type=='test_generation':
          self.z_s,self.z_sg,self.z_g = self.get_random_z()
          self.generated_adj, self.generated_adj_prob, self.generated_spatial,  self.generated_node_feat= self.sample(self.z_s,self.z_sg,self.z_g)
        if FLAGS.type=='test_disentangle':
           self.generated_adj, self.generated_adj_prob, self.generated_spatial,  self.generated_node_feat= self.traverse_generation(self.group_type,self.dim_a,self.dim_b,self.dim_c)
        if FLAGS.type=='sample':
            self.z_s,self.z_sg,self.z_g = self.get_random_z()
            self.generated_adj, self.generated_adj_prob, self.generated_spatial,  self.generated_node_feat= self.sample(self.z_s,self.z_sg,self.z_g)

        t_vars = tf.trainable_variables()

        self.vars = [var for var in t_vars]
        self.saver = tf.train.Saver()


    def encoder(self):
        with tf.variable_scope("encoder") as scope:
            FLAGS.sg_batch_size = int(FLAGS.sg_batch_size * FLAGS.sampling_num)
            FLAGS.sg_decoder_batch_size = int(FLAGS.sg_decoder_batch_size * FLAGS.sampling_num)
            print (self.node_feature.shape, self.inputs_3d.shape, self.rel.shape)
            #graph entangled embeddings:
            g=tf.reshape(self.node_feature,[FLAGS.batch_size,self.n_samples,-1])
            for i in range(FLAGS.graph_conv_layers): # 2
              name_='g_g'+str(i)+'_conv'
              g=self.g_bn_g[i](GraphConvolution(self.adj, g, FLAGS.g_conv_hidden[i], name=name_))
            #   g = tf.nn.dropout(g, self.dropout)
              g=tf.concat([g,self.node_feature],axis=-1)
            #g is B*N*final_hidden
            # symmetry
            g=self.encoder_g(g)
            g_=linear(tf.reshape(g, [FLAGS.batch_size, -1]), FLAGS.g_hidden_size, name='g_g1_lin')
            self.z_mean_g=linear(tf.reshape(g_, [FLAGS.batch_size, -1]), FLAGS.g_latent_size, name='g_g2_lin')
            self.z_std_g=linear(tf.reshape(g_, [FLAGS.batch_size, -1]), FLAGS.g_latent_size, name='g_g3_lin')


            #spatial info embeddings:
            h = tf.reshape(self.inputs_3d, [FLAGS.batch_size,self.n_samples,FLAGS.spatial_dim])
            for i in range(FLAGS.spatial_conv_layers): # 5
                name_ = 'g_s'+str(i+1)+'_conv'
                h = self.g_bn_s[i](tf.layers.conv1d(h, FLAGS.s_channel[i], FLAGS.s_kernel_size[i], FLAGS.s_strides[i],name=name_, padding='SAME'))
                h = tf.nn.relu(h)
            #h is B*~*final_channel
            # symmetry
            h=self.encoder_s(h) 
            h_=linear(tf.reshape(h, [FLAGS.batch_size, -1]), FLAGS.s_hidden_size, name='g_s1_lin')
            self.z_mean_s=linear(tf.reshape(h_, [FLAGS.batch_size, -1]), FLAGS.s_latent_size, 'g_s2_lin')
            self.z_std_s=linear(tf.reshape(h_, [FLAGS.batch_size, -1]), FLAGS.s_latent_size, 'g_s3_lin')



            # spatial-network joint embeddings:
            s_g=self.node_feature_sg
            for i in range(FLAGS.spatial_graph_conv_layers):
                name_='g_sg'+str(i)+'_conv'
                if FLAGS.dataset == 'synthetic1' or FLAGS.dataset == 'synthetic2' or FLAGS.dataset == 'synthetic3':
                    s_g=self.g_bn_sg[i](SpatialGraphConvolution(self.adj_sg, s_g, self.rel_sg, FLAGS.sg_conv_hidden[i], name=name_))
                elif FLAGS.dataset == 'protein' or FLAGS.dataset == 'mnist':
                    s_g=self.g_bn_sg[i](SpatialGraphConvolution_3D(self.adj_sg, s_g, self.rel_sg, FLAGS.sg_conv_hidden[i], name=name_))
                elif FLAGS.model_type == 'geoGCN':
                    s_g=self.g_bn_sg[i](GeoGraphConvolution_adj_layer0(self.adj, s_g, self.rel, FLAGS.sg_conv_hidden[i], name=name_))
                elif FLAGS.model_type == 'posGCN':
                    s_g=self.g_bn_sg[i](StructGraphConvolution_adj_layer0(self.adj, s_g, self.inputs_3d, FLAGS.sg_conv_hidden[i], name=name_))

                s_g=lrelu(s_g)
            # symmetry
            s_g=self.encoder_sg(s_g)
            s_g_=linear(tf.reshape(s_g, [FLAGS.sg_batch_size, -1]), FLAGS.sg_hidden_size, name='g_sg1_lin')
            self.z_mean_sg=linear(tf.reshape(s_g_, [FLAGS.sg_batch_size, -1]), FLAGS.sg_latent_size, name='g_sg2_lin')
            self.z_std_sg=linear(tf.reshape(s_g_, [FLAGS.sg_batch_size, -1]), FLAGS.sg_latent_size, name='g_sg3_lin')

    def get_z(self, random):

        z_s=self.z_mean_s+ tf.random.normal([FLAGS.batch_size,FLAGS.s_latent_size]) * tf.exp(self.z_std_s)

        z_sg=self.z_mean_sg+ tf.random.normal([FLAGS.sg_batch_size,FLAGS.sg_latent_size]) * tf.exp(self.z_std_sg)

        z_g= self.z_mean_g+ tf.random.normal([FLAGS.batch_size,FLAGS.g_latent_size]) * tf.exp(self.z_std_g)

        return z_s,z_sg,z_g

    def get_random_z(self):

        z_s = tf.random.normal([FLAGS.batch_size,FLAGS.s_latent_size])
        z_sg = tf.random.normal([FLAGS.sg_batch_size,FLAGS.sg_latent_size])
        z_g = tf.random.normal([FLAGS.batch_size,FLAGS.g_latent_size])

        return z_s,z_sg,z_g


    def decoder(self, z_s, z_sg, z_g):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as scope:
            FLAGS.sg_batch_size = int(FLAGS.sg_batch_size/FLAGS.sampling_num)
            FLAGS.sg_decoder_batch_size = int(FLAGS.sg_decoder_batch_size/FLAGS.sampling_num)
            print (z_sg.shape, z_s.shape)
            z_sg = tf.reshape(linear(z_sg, self.n_samples*FLAGS.node_h_size, name='d_sg_lin1'),[FLAGS.sg_decoder_batch_size,FLAGS.sampling_num,self.n_samples,FLAGS.node_h_size])
            z_s = tf.reshape(linear(z_s, self.n_samples*FLAGS.node_h_size, name='d_s_lin1'),[FLAGS.decoder_batch_size,self.n_samples,FLAGS.node_h_size])
            z_g = tf.reshape(linear(z_g, self.n_samples*FLAGS.node_h_size, name='d_g_lin1'),[FLAGS.decoder_batch_size,self.n_samples,FLAGS.node_h_size])
            z_sg = tf.reduce_mean(z_sg,axis=1)
            # z_s = tf.reduce_mean(z_s,axis=1)
            # z_g = tf.reduce_mean(z_g,axis=1)

            #decoing the graph:
            diag=np.tile(np.ones(self.n_samples),[FLAGS.decoder_batch_size,1,1])-np.tile(np.eye(self.n_samples),[FLAGS.decoder_batch_size,1,1])
            z_sg_g=tf.concat((z_sg,z_g),axis=-1)
            #generate node addtributes
            node_z_n=z_sg_g
            for i in range(FLAGS.graph_deconv_layers):
                name_='n'+str(i)+'_deconv'
                node_z_n=self.d_bn_n[i](tf.layers.conv1d(node_z_n, FLAGS.n_d_channel[i], FLAGS.n_d_kernel_size[i], FLAGS.n_d_strides[i], name=name_, padding='SAME'))
                # node_z_n = tf.nn.dropout(lrelu(node_z_n), self.dropout)
            generated_node_feat=tf.nn.sigmoid(linear(self.decoder_node(tf.reshape(node_z_n, [FLAGS.batch_size*self.n_samples, -1])), FLAGS.num_feature, name='d_n_lin2'))
            generated_node_feat=tf.reshape(generated_node_feat,[FLAGS.batch_size, self.n_samples, -1])
            #generate adj
            adj_z_n1=tf.reshape(z_sg_g,[FLAGS.decoder_batch_size,self.n_samples,1, -1]) #tf.reshape(tf.concat((graph_h, joint_h),axis=-1),[FLAGS.decoder_batch_size,self.n_samples,1,-1])
            adj_z_n2=tf.reshape(z_sg_g,[FLAGS.decoder_batch_size,1, self.n_samples, -1])
            adj_z_n=tf.concat([tf.tile(adj_z_n1,[1,1,self.n_samples,1]),tf.tile(adj_z_n2,[1,self.n_samples,1,1])],axis=-1)
            for i in range(FLAGS.graph_deconv_layers):
                name_='e'+str(i)+'_deconv'
                adj_z_n = self.d_bn_e[i](adj_z_n)
                adj_z_n= e2e(tf.nn.relu(adj_z_n),FLAGS.e_d_hidden[i],k_h=self.n_samples, name=name_)
            generated_adj_prob_origin=linear(tf.reshape(tf.nn.relu(self.decoder_adj(adj_z_n)),[FLAGS.decoder_batch_size*self.n_samples*self.n_samples,-1]),2, name='d_e_lin2')#B*N*N*2
            #remove the diag
            generated_adj_prob1=diag*tf.reshape(generated_adj_prob_origin,[FLAGS.decoder_batch_size, self.n_samples, self.n_samples,2])[:,:,:,1]
            generated_adj_prob0=diag*tf.reshape(generated_adj_prob_origin,[FLAGS.decoder_batch_size, self.n_samples, self.n_samples,2])[:,:,:,0]+(1-diag)
            generated_adj_prob=tf.concat([tf.reshape(generated_adj_prob0,[FLAGS.decoder_batch_size, self.n_samples, self.n_samples,1]),tf.reshape(generated_adj_prob1,[FLAGS.decoder_batch_size, self.n_samples, self.n_samples,1])],axis=-1)
            generated_adj= tf.argmax(tf.nn.softmax(generated_adj_prob,axis=-1),axis=-1) #B*N*N



            #decoding the spatial information:
            spatial_h=tf.concat((z_sg, z_s),axis=-1)
            for i in range(FLAGS.spatial_deconv_layers): # 5
                name = 's'+str(i+1)+'_deconv'
                spatial_h = self.d_bn_s[i](tf.layers.conv1d(spatial_h, FLAGS.s_d_channel[i], FLAGS.s_d_kernel_size[i], FLAGS.s_d_strides[i], name=name, padding='SAME'))
                # spatial_h = tf.nn.dropout(lrelu(spatial_h), self.dropout)
            generated_spatial=tf.nn.sigmoid(linear(tf.reshape(spatial_h, [FLAGS.batch_size*self.n_samples, -1]), FLAGS.spatial_dim, name='d_s_lin2'))
            generated_spatial=tf.reshape(generated_spatial,[FLAGS.batch_size, self.n_samples, -1])


            return generated_adj, generated_adj_prob, generated_spatial,  generated_node_feat




    def sample(self, z_s, z_sg, z_g):
            generated_adj, generated_adj_prob, spatial,  generated_node_feat= self.decoder(z_s,z_sg,z_g)
            return generated_adj, generated_adj_prob, spatial,  generated_node_feat


    def traverse(self,group_type,fix_dim):
            #make one dimension of one node changed and other fixed
            length=FLAGS.g_latent_size+FLAGS.s_latent_size+FLAGS.sg_latent_size
            z_s=np.load('./qualitative_evaluation/'+str(FLAGS.dataset)+"/"+FLAGS.vae_type+'_z_s.npy').reshape(-1,1,FLAGS.s_latent_size)[1*length:2*length]#+ np.random.normal(1,0.1,[length,1,FLAGS.s_latent_size]) * np.exp(0.1)
            z_g=np.load('./qualitative_evaluation/'+str(FLAGS.dataset)+"/"+FLAGS.vae_type+'_z_g.npy').reshape(-1,1,FLAGS.g_latent_size)[1*length:2*length]#+ np.random.normal(1,0.1,[length,1,FLAGS.g_latent_size]) * np.exp(0.1)
            z_sg= np.load('./qualitative_evaluation/'+str(FLAGS.dataset)+"/"+FLAGS.vae_type+'_z_sg.npy').reshape(-1,1,FLAGS.sg_latent_size)[1*length:2*length]#+ np.random.normal(1,0.1,[length,1,FLAGS.sg_latent_size]) * np.exp(0.1)

            z_s=np.tile(z_s,[1,FLAGS.visualize_length,1]).reshape(-1,FLAGS.s_latent_size)
            z_g=np.tile(z_g,[1,FLAGS.visualize_length,1]).reshape(-1,FLAGS.g_latent_size)
            z_sg=np.tile(z_sg,[1,FLAGS.visualize_length,1]).reshape(-1,FLAGS.sg_latent_size)
            rang= np.arange(-4,4,0.8)[:FLAGS.visualize_length]


            if group_type =='s':
                base=0
                rang = np.arange(-100,20,4)[:FLAGS.visualize_length]
                z_s[fix_dim*FLAGS.visualize_length+base:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length+base,fix_dim]=rang
            elif group_type == 'g':
                rang= np.arange(-60,60,4)[:FLAGS.visualize_length]
                base=FLAGS.s_latent_size*FLAGS.visualize_length
                z_g[fix_dim*FLAGS.visualize_length+base:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length+base,fix_dim]=rang
            elif group_type == 'sg':
                rang= np.arange(-30,30,2)[:FLAGS.visualize_length]
                base=FLAGS.s_latent_size*FLAGS.visualize_length+FLAGS.g_latent_size*FLAGS.visualize_length
                z_sg[fix_dim*FLAGS.visualize_length+base:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length+base,fix_dim]=rang

            z_s=z_s[fix_dim*FLAGS.visualize_length+base:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length+base]
            z_g=z_g[fix_dim*FLAGS.visualize_length+base:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length+base]
            z_sg=z_sg[fix_dim*FLAGS.visualize_length+base:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length+base]
            FLAGS.decoder_batch_size = FLAGS.visualize_length

            generated_adj, generated_adj_prob, spatial,  generated_node_feat= self.decoder(tf.convert_to_tensor(z_s, 'float32'),tf.convert_to_tensor(z_sg,'float32'),tf.convert_to_tensor(z_g, 'float32'))

            return generated_adj, generated_adj_prob, spatial,  generated_node_feat

    def traverse_generation(self,group_type,fix_dim_a,fix_dim_b,fix_dim_c):
            #make one dimension of one node changed and other fixed
            length=FLAGS.g_latent_size+FLAGS.s_latent_size+FLAGS.sg_latent_size
            z_s=np.load('./qualitative_evaluation/'+str(FLAGS.dataset)+"/"+FLAGS.vae_type+'_z_s.npy').reshape(-1,1,FLAGS.s_latent_size)[1*length:2*length]#+ np.random.normal(1,0.1,[length,1,FLAGS.s_latent_size]) * np.exp(0.1)
            z_g=np.load('./qualitative_evaluation/'+str(FLAGS.dataset)+"/"+FLAGS.vae_type+'_z_g.npy').reshape(-1,1,FLAGS.g_latent_size)[1*length:2*length]#+ np.random.normal(1,0.1,[length,1,FLAGS.g_latent_size]) * np.exp(0.1)
            z_sg= np.load('./qualitative_evaluation/'+str(FLAGS.dataset)+"/"+FLAGS.vae_type+'_z_sg.npy').reshape(-1,1,FLAGS.sg_latent_size)[1*length:2*length]#+ np.random.normal(1,0.1,[length,1,FLAGS.sg_latent_size]) * np.exp(0.1)

            z_s=np.tile(z_s,[1,FLAGS.visualize_length,1]).reshape(-1,FLAGS.s_latent_size)
            z_g=np.tile(z_g,[1,FLAGS.visualize_length,1]).reshape(-1,FLAGS.g_latent_size)
            z_sg=np.tile(z_sg,[1,FLAGS.visualize_length,1]).reshape(-1,FLAGS.sg_latent_size)
            rang= np.arange(-4,4,0.8)[:FLAGS.visualize_length]

            z_sg_cp = np.copy(z_sg)

            base=0
            # rang = np.arange(-20,20,2)[:FLAGS.visualize_length] # for synthetic1
            rang = np.arange(-20,20,2)[:FLAGS.visualize_length] # for synthetic2
            z_s[fix_dim_a*FLAGS.visualize_length+base:fix_dim_a*FLAGS.visualize_length+FLAGS.visualize_length+base,fix_dim_a]=rang
            rang= np.arange(-1,1,0.1)[:FLAGS.visualize_length]
            base=FLAGS.s_latent_size*FLAGS.visualize_length
            z_g[fix_dim_b*FLAGS.visualize_length+base:fix_dim_b*FLAGS.visualize_length+FLAGS.visualize_length+base,fix_dim_b]=rang
            rang= np.arange(-10,10,1)[:FLAGS.visualize_length]
            base=FLAGS.s_latent_size*FLAGS.visualize_length+FLAGS.g_latent_size*FLAGS.visualize_length
            z_sg[fix_dim_c*FLAGS.visualize_length+base:fix_dim_c*FLAGS.visualize_length+FLAGS.visualize_length+base,fix_dim_c]=rang

            # synthetic2 visualziation parameter
            z_g1=z_g[fix_dim_c*FLAGS.visualize_length+base:fix_dim_c*FLAGS.visualize_length+FLAGS.visualize_length+base]
            base = 0
            z_s1=z_s[fix_dim_a*FLAGS.visualize_length+base:fix_dim_a*FLAGS.visualize_length+FLAGS.visualize_length+base]
            z_sg1=z_sg[fix_dim_a*FLAGS.visualize_length+base:fix_dim_a*FLAGS.visualize_length+FLAGS.visualize_length+base]
            base = FLAGS.s_latent_size*FLAGS.visualize_length+FLAGS.g_latent_size*FLAGS.visualize_length
            z_s1=np.concatenate((z_s1,z_s[fix_dim_c*FLAGS.visualize_length+base:fix_dim_c*FLAGS.visualize_length+FLAGS.visualize_length+base]))
            base = FLAGS.s_latent_size*FLAGS.visualize_length
            z_sg1=np.concatenate((z_sg1,z_sg[fix_dim_b*FLAGS.visualize_length+base:fix_dim_b*FLAGS.visualize_length+FLAGS.visualize_length+base]))
            z_g1=np.concatenate((z_g1,z_g[fix_dim_b*FLAGS.visualize_length+base:fix_dim_b*FLAGS.visualize_length+FLAGS.visualize_length+base]))
            base = FLAGS.s_latent_size*FLAGS.visualize_length+FLAGS.g_latent_size*FLAGS.visualize_length
            z_s1=np.concatenate((z_s1,z_s[fix_dim_c*FLAGS.visualize_length+base:fix_dim_c*FLAGS.visualize_length+FLAGS.visualize_length+base]))
            z_sg1=np.concatenate((z_sg1,z_sg[fix_dim_c*FLAGS.visualize_length+base:fix_dim_c*FLAGS.visualize_length+FLAGS.visualize_length+base]))
            z_g1=np.concatenate((z_g1,z_g[fix_dim_c*FLAGS.visualize_length+base:fix_dim_c*FLAGS.visualize_length+FLAGS.visualize_length+base]))

            # synthetic1 visualization parameter
            # base = 0
            # z_s1=z_s[fix_dim_a*FLAGS.visualize_length+base:fix_dim_a*FLAGS.visualize_length+FLAGS.visualize_length+base]
            # z_g1=z_g[fix_dim_a*FLAGS.visualize_length+base:fix_dim_a*FLAGS.visualize_length+FLAGS.visualize_length+base]
            # z_sg1=z_sg[fix_dim_a*FLAGS.visualize_length+base:fix_dim_a*FLAGS.visualize_length+FLAGS.visualize_length+base]
            # base = FLAGS.s_latent_size*FLAGS.visualize_length+FLAGS.g_latent_size*FLAGS.visualize_length
            # z_sg1=np.concatenate((z_sg1,z_sg_cp[fix_dim_c*FLAGS.visualize_length+base:fix_dim_c*FLAGS.visualize_length+FLAGS.visualize_length+base]))
            # z_s1=np.concatenate((z_s1,z_s[fix_dim_c*FLAGS.visualize_length+base:fix_dim_c*FLAGS.visualize_length+FLAGS.visualize_length+base]))
            # base = FLAGS.s_latent_size*FLAGS.visualize_length
            # z_g1=np.concatenate((z_g1,z_g[fix_dim_b*FLAGS.visualize_length+base:fix_dim_b*FLAGS.visualize_length+FLAGS.visualize_length+base]))
            # base = FLAGS.s_latent_size*FLAGS.visualize_length+FLAGS.g_latent_size*FLAGS.visualize_length
            # z_s1=np.concatenate((z_s1,z_s[fix_dim_c*FLAGS.visualize_length+base:fix_dim_c*FLAGS.visualize_length+FLAGS.visualize_length+base]))
            # z_sg1=np.concatenate((z_sg1,z_sg[fix_dim_c*FLAGS.visualize_length+base:fix_dim_c*FLAGS.visualize_length+FLAGS.visualize_length+base]))
            # z_g1=np.concatenate((z_g1,z_g[fix_dim_c*FLAGS.visualize_length+base:fix_dim_c*FLAGS.visualize_length+FLAGS.visualize_length+base]))
            FLAGS.decoder_batch_size = FLAGS.visualize_length*3
            print (z_s1.shape, z_sg1.shape, z_g1.shape)
            generated_adj, generated_adj_prob, spatial,  generated_node_feat= self.decoder(tf.convert_to_tensor(z_s1, 'float32'),tf.convert_to_tensor(z_sg1,'float32'),tf.convert_to_tensor(z_g1, 'float32'))
            return generated_adj, generated_adj_prob, spatial,  generated_node_feat

    def traverse_latent(self,group_type,fix_dim):
            #make one dimension of one node changed and other fixed
            length=FLAGS.g_latent_size+FLAGS.s_latent_size+FLAGS.sg_latent_size
            z_s=np.load('./qualitative_evaluation/'+str(FLAGS.dataset)+"/"+FLAGS.vae_type+'_z_s.npy').reshape(-1,1,FLAGS.s_latent_size)[1*length:2*length]#+ np.random.normal(1,0.1,[length,1,FLAGS.s_latent_size]) * np.exp(0.1)
            z_g=np.load('./qualitative_evaluation/'+str(FLAGS.dataset)+"/"+FLAGS.vae_type+'_z_g.npy').reshape(-1,1,FLAGS.g_latent_size)[1*length:2*length]#+ np.random.normal(1,0.1,[length,1,FLAGS.g_latent_size]) * np.exp(0.1)
            z_sg= np.load('./qualitative_evaluation/'+str(FLAGS.dataset)+"/"+FLAGS.vae_type+'_z_sg.npy').reshape(-1,1,FLAGS.sg_latent_size)[1*length:2*length]#+ np.random.normal(1,0.1,[length,1,FLAGS.sg_latent_size]) * np.exp(0.1)

            z_s=np.tile(z_s,[1,FLAGS.visualize_length,1]).reshape(-1,FLAGS.s_latent_size)
            z_g=np.tile(z_g,[1,FLAGS.visualize_length,1]).reshape(-1,FLAGS.g_latent_size)
            z_sg=np.tile(z_sg,[1,FLAGS.visualize_length,1]).reshape(-1,FLAGS.sg_latent_size)
            rang= np.arange(-2,2,0.2)[:FLAGS.visualize_length]


            for dim in range(FLAGS.s_latent_size):
                base=0
                rang = np.arange(-10,10,2)[:FLAGS.visualize_length]
                z_s[dim*FLAGS.visualize_length+base:dim*FLAGS.visualize_length+FLAGS.visualize_length+base,dim]=rang
            for dim in range(FLAGS.g_latent_size):
                rang = np.arange(-10,10,2)[:FLAGS.visualize_length]
                base=FLAGS.s_latent_size*FLAGS.visualize_length
                z_g[dim*FLAGS.visualize_length+base:dim*FLAGS.visualize_length+FLAGS.visualize_length+base,dim]=rang
            for dim in range(FLAGS.sg_latent_size):
                base=FLAGS.s_latent_size*FLAGS.visualize_length+FLAGS.g_latent_size*FLAGS.visualize_length
                z_sg[dim*FLAGS.visualize_length+base:dim*FLAGS.visualize_length+FLAGS.visualize_length+base,dim]=rang

            # z_s=z_s[fix_dim*FLAGS.visualize_length+base:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length+base]
            # z_g=z_g[fix_dim*FLAGS.visualize_length+base:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length+base]
            # z_sg=z_sg[fix_dim*FLAGS.visualize_length+base:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length+base]
            # FLAGS.decoder_batch_size = FLAGS.visualize_length

            generated_adj, generated_adj_prob, spatial,  generated_node_feat= self.decoder(tf.convert_to_tensor(z_s, 'float32'),tf.convert_to_tensor(z_sg,'float32'),tf.convert_to_tensor(z_g, 'float32'))

            return generated_adj, generated_adj_prob, spatial,  generated_node_feat

            #for fix_dim in range(FLAGS.s_latent_size):
             # z_s[fix_dim*FLAGS.visualize_length:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length,fix_dim]=rang
            #for fix_dim in range(FLAGS.g_latent_size):
            #  base=FLAGS.s_latent_size*FLAGS.visualize_length
            #  z_g[fix_dim*FLAGS.visualize_length+base:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length+base,fix_dim]=rang
            #for fix_dim in range(FLAGS.sg_latent_size):
            #  base=FLAGS.s_latent_size*FLAGS.visualize_length+FLAGS.g_latent_size*FLAGS.visualize_length
            #  z_sg[fix_dim*FLAGS.visualize_length+base:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length+base,fix_dim]=rang


            #generated_adj, generated_adj_prob, spatial,  generated_node_feat= self.decoder(tf.convert_to_tensor(z_s, 'float32'),tf.convert_to_tensor(z_sg,'float32'),tf.convert_to_tensor(z_g, 'float32'))
