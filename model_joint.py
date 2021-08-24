from layers import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

flags = tf.app.flags
FLAGS = flags.FLAGS




class SGCNModelVAE(object):
        
    '''VGAE Model for reconstructing graph edges from node representations.'''
    def __init__(self, placeholders, num_features, num_nodes, **kwargs):
        super(SGCNModelVAE, self).__init__(**kwargs)
        self.node_feature = placeholders['features']  #node attributes B*N*h
        self.inputs_3d = placeholders['spatial']  #B*N*3
        self.input_dim = num_features
        self.n_samples = num_nodes
        self.adj = placeholders['adj']   #B*N*N (element [i,j] refers to whether ane edge between i  and j )
        self.rel = placeholders['rel']   #B*N*N (element [i,j] refers to spatial distance between i  and j )
        self.dropout = placeholders['dropout']
        self.weight_norm = 0       
        
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
 
        self.d_bn_n=[]
        self.d_bn_e=[]
        for i in range(FLAGS.graph_deconv_layers):
          self.d_bn_n.append(batch_norm(name='d_bn_n'+str(i))) 
          self.d_bn_e.append(batch_norm(name='d_bn_e'+str(i))) 
          
        self.d_bn_s=[]
        for i in range(FLAGS.spatial_deconv_layers):
          self.d_bn_s.append(batch_norm(name='d_bn_s'+str(i))) 

          
        self._build()
        
    def _build(self):
        self.encoder()
        self.z_sg = self.get_z(random = True)
        #z_noiseless = self.get_z(random = False) 
        if FLAGS.type=='train':
          self.generated_adj, self.generated_adj_prob, self.generated_spatial,  self.generated_node_feat = self.decoder(self.z_sg)
        if FLAGS.type=='test_reconstruct': 
          self.generated_adj, self.generated_adj_prob, self.generated_spatial,  self.generated_node_feat= self.sample(self.z_sg)
        if FLAGS.type == 'test_generation':
            self.z_sg = self.get_random_z()
            self.generated_adj, self.generated_adj_prob, self.generated_spatial,  self.generated_node_feat= self.sample(self.z_sg)                    
        if FLAGS.type=='test_disentangle': 
          self.generated_adj, self.generated_adj_prob, self.generated_spatial,  self.generated_node_feat= self.traverse(FLAGS.group_type,FLAGS.dim)
         
        t_vars = tf.trainable_variables()

        self.vars = [var for var in t_vars]
        self.saver = tf.train.Saver()
     
        
    def encoder(self):
        with tf.variable_scope("encoder") as scope:

            # spatial-network joint embeddings: 
            s_g=self.node_feature              
            for i in range(FLAGS.spatial_graph_conv_layers):
                name_='g_sg'+str(i)+'_conv' 
                s_g=self.g_bn_sg[i](SpatialGraphConvolution(self.adj, s_g, self.rel, FLAGS.sg_conv_hidden[i], name=name_))
                s_g=lrelu(s_g)
                s_g=tf.nn.dropout(s_g, rate=1-self.dropout)
            # symmetry             
            s_g_=linear(tf.reshape(s_g, [FLAGS.batch_size, -1]), FLAGS.sg_hidden_size, name='g_sg1_lin')
            self.z_mean_sg=linear(tf.reshape(s_g_, [FLAGS.batch_size, -1]), FLAGS.sg_latent_size, name='g_sg2_lin')
            self.z_std_sg=linear(tf.reshape(s_g_, [FLAGS.batch_size, -1]), FLAGS.sg_latent_size, name='g_sg3_lin')

    def get_z(self, random):
        
        z_sg=self.z_mean_sg+ tf.random.normal([FLAGS.batch_size, FLAGS.sg_latent_size]) * tf.exp(self.z_std_sg)
                    
        return z_sg
        
      
    def decoder(self, z_sg):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as scope:
                       
            joint_h = tf.reshape(linear(z_sg, self.n_samples*FLAGS.node_h_size, name='d_sg_lin1'),[FLAGS.decoder_batch_size,self.n_samples,FLAGS.node_h_size])                                                         
            
            
                       
            #decoding the spatial information:
            #spatial_h_nodes=joint_h       
            #spatial_z_n1=tf.reshape(joint_h,[FLAGS.decoder_batch_size,self.n_samples,1, -1]) #tf.reshape(tf.concat((graph_h, joint_h),axis=-1),[FLAGS.decoder_batch_size,self.n_samples,1,-1]) 
            #spatial_z_n2=tf.reshape(joint_h,[FLAGS.decoder_batch_size,1, self.n_samples, -1])
            #spatial_z_n=tf.concat([tf.tile(spatial_z_n1,[1,1,self.n_samples,1]),tf.tile(spatial_z_n2,[1,self.n_samples,1,1])],axis=-1)
            #for i in range(FLAGS.graph_deconv_layers):
            #    name_='e'+str(i)+'_deconv'  
            #    spatial_z_n = self.d_bn_s[i](spatial_z_n) 
            #    spatial_z_n= e2e(tf.nn.relu(spatial_z_n),FLAGS.e_d_hidden[i],k_h=self.n_samples, name=name_)                    
            #generated_rel=linear(tf.reshape(tf.nn.relu(spatial_z_n),[FLAGS.decoder_batch_size*self.n_samples*self.n_samples,-1]),1, name='d_s_lin2')#B*N*N*1              
            #generated_rel=tf.reshape(generated_rel,[FLAGS.decoder_batch_size, self.n_samples, self.n_samples, 1])
            spatial_h=joint_h                     
            for i in range(FLAGS.spatial_deconv_layers): # 5          
                name = 's'+str(i+1)+'_deconv'
                spatial_h = self.d_bn_s[i](tf.layers.conv1d(spatial_h, FLAGS.s_d_channel[i], FLAGS.s_d_kernel_size[i], FLAGS.s_d_strides[i], name=name, padding='SAME'))
                spatial_h = tf.nn.dropout(lrelu(spatial_h), self.dropout)
                
            if FLAGS.dataset in ['synthetic3','scene']: 
                generated_spatial=linear(tf.reshape(spatial_h, [FLAGS.batch_size*self.n_samples, -1]), FLAGS.spatial_dim, name='d_s_lin2')
            else:    
                generated_spatial=tf.nn.sigmoid(linear(tf.reshape(spatial_h, [FLAGS.batch_size*self.n_samples, -1]), FLAGS.spatial_dim, name='d_s_lin2'))
            
            generated_spatial=tf.reshape(generated_spatial,[FLAGS.batch_size, self.n_samples, -1])              
            
            #decoing the graph:
            diag=np.tile(np.ones(self.n_samples),[FLAGS.decoder_batch_size,1,1])-np.tile(np.eye(self.n_samples),[FLAGS.decoder_batch_size,1,1])                                                
            #generate node addtributes 
            
            node_z_n=joint_h  
            #for i in range(FLAGS.graph_deconv_layers):      
            #    name_='n'+str(i)+'_deconv'
            #    node_z_n=self.d_bn_n[i](tf.layers.conv1d(node_z_n, FLAGS.n_d_channel[i], FLAGS.n_d_kernel_size[i], FLAGS.n_d_strides[i], name=name_, padding='SAME'))  
            #    node_z_n = tf.nn.dropout(lrelu(node_z_n), self.dropout)
            #generated_node_feat=tf.nn.sigmoid(linear(tf.reshape(node_z_n, [FLAGS.batch_size*self.n_samples, -1]), FLAGS.num_feature, name='d_n_lin2'))
            #generated_node_feat=tf.reshape(generated_node_feat,[FLAGS.batch_size, self.n_samples, -1])   
            for i in range(FLAGS.graph_deconv_layers):      
                name_='n'+str(i)+'_deconv'
                node_z_n=self.d_bn_n[i](tf.layers.conv1d(node_z_n, FLAGS.n_d_channel[i], FLAGS.n_d_kernel_size[i], FLAGS.n_d_strides[i], name=name_, padding='SAME'))  
                node_z_n = tf.nn.dropout(lrelu(node_z_n), self.dropout)
            if FLAGS.dataset == 'scene':
               self.generated_node_feat_prob=linear(tf.reshape(node_z_n, [FLAGS.batch_size*self.n_samples, -1]), FLAGS.num_feature, name='d_n_lin2')                           
               generated_node_feat=tf.argmax(tf.nn.softmax(self.generated_node_feat_prob,axis=-1),axis=-1)
            else:
               generated_node_feat=tf.nn.sigmoid(linear(tf.reshape(node_z_n, [FLAGS.batch_size*self.n_samples, -1]), FLAGS.num_feature, name='d_n_lin2'))
               generated_node_feat=tf.reshape(generated_node_feat,[FLAGS.batch_size, self.n_samples, -1]) 


                  
            #generate adj   
            #adj_z_n1=tf.reshape(joint_h,[FLAGS.decoder_batch_size,self.n_samples,1, -1]) #tf.reshape(tf.concat((graph_h, joint_h),axis=-1),[FLAGS.decoder_batch_size,self.n_samples,1,-1]) 
            #adj_z_n2=tf.reshape(joint_h,[FLAGS.decoder_batch_size,1, self.n_samples, -1])
            #adj_z_n=tf.concat([tf.tile(adj_z_n1,[1,1,self.n_samples,1]),tf.tile(adj_z_n2,[1,self.n_samples,1,1])],axis=-1)
            #for i in range(FLAGS.graph_deconv_layers):
             #   name_='e'+str(i)+'_deconv'  
            #    adj_z_n = self.d_bn_e[i](adj_z_n) 
            #    adj_z_n= e2e(tf.nn.relu(adj_z_n),FLAGS.e_d_hidden[i],k_h=self.n_samples, name=name_)            
             
            #generated_adj_prob_origin=linear(tf.reshape(tf.nn.relu(adj_z_n),[FLAGS.decoder_batch_size*self.n_samples*self.n_samples,-1]),2, name='d_e_lin2')#B*N*N*2  
            #remove the diag
            #generated_adj_prob1=diag*tf.reshape(generated_adj_prob_origin,[FLAGS.decoder_batch_size, self.n_samples, self.n_samples,2])[:,:,:,1] 
            #generated_adj_prob0=diag*tf.reshape(generated_adj_prob_origin,[FLAGS.decoder_batch_size, self.n_samples, self.n_samples,2])[:,:,:,0]+(1-diag) 
            #generated_adj_prob=tf.concat([tf.reshape(generated_adj_prob0,[FLAGS.decoder_batch_size, self.n_samples, self.n_samples,1]),tf.reshape(generated_adj_prob1,[FLAGS.decoder_batch_size, self.n_samples, self.n_samples,1])],axis=-1)
            #generated_adj= tf.argmax(tf.nn.softmax(generated_adj_prob,axis=-1),axis=-1) #B*N*N
            adj_z_n1=tf.reshape(joint_h,[FLAGS.decoder_batch_size,self.n_samples,1, -1]) #tf.reshape(tf.concat((graph_h, joint_h),axis=-1),[FLAGS.decoder_batch_size,self.n_samples,1,-1]) 
            adj_z_n2=tf.reshape(joint_h,[FLAGS.decoder_batch_size,1, self.n_samples, -1])
            adj_z_n=tf.concat([tf.tile(adj_z_n1,[1,1,self.n_samples,1]),tf.tile(adj_z_n2,[1,self.n_samples,1,1])],axis=-1)
            for i in range(FLAGS.graph_deconv_layers):
                name_='e'+str(i)+'_deconv'  
                adj_z_n = self.d_bn_e[i](adj_z_n) 
                adj_z_n= e2e(tf.nn.relu(adj_z_n),FLAGS.e_d_hidden[i],k_h=self.n_samples, name=name_)                         
            generated_adj_prob_origin=linear(tf.reshape(tf.nn.relu(adj_z_n),[FLAGS.decoder_batch_size*self.n_samples*self.n_samples,-1]),FLAGS.num_edge_feature, name='d_e_lin2')#B*N*N*2              
            #remove the diag          
            if FLAGS.dataset == 'scene':
                generated_adj_prob=generated_adj_prob_origin
            else:
                generated_adj_prob1=diag*tf.reshape(generated_adj_prob_origin,[FLAGS.decoder_batch_size, self.n_samples, self.n_samples,FLAGS.num_edge_feature])[:,:,:,1] 
                generated_adj_prob0=diag*tf.reshape(generated_adj_prob_origin,[FLAGS.decoder_batch_size, self.n_samples, self.n_samples,FLAGS.num_edge_feature])[:,:,:,0]+(1-diag)                 
                generated_adj_prob=tf.concat([tf.reshape(generated_adj_prob0,[FLAGS.decoder_batch_size, self.n_samples, self.n_samples,1]),tf.reshape(generated_adj_prob1,[FLAGS.decoder_batch_size, self.n_samples, self.n_samples,1])],axis=-1)                
            generated_adj= tf.argmax(tf.nn.softmax(generated_adj_prob,axis=-1),axis=-1) #B*N*N
                        
 
            return generated_adj, generated_adj_prob, generated_spatial,  generated_node_feat




    def sample(self, z_sg):
        
        generated_adj, generated_adj_prob, spatial,  generated_node_feat= self.decoder(z_sg)  
        return generated_adj, generated_adj_prob, spatial,  generated_node_feat
            
    def traverse(self, group_name, fix_dim):        
                       
            #make one dimension of one node changed and other fixed 
            length=FLAGS.sg_latent_size
            z_sg= np.load('./qualitative_evaluation/'+FLAGS.vae_type+'_z_sg.npy').reshape(-1,1,FLAGS.sg_latent_size)[1*length:2*length]#+ np.random.normal(1,0.1,[length,1,FLAGS.sg_latent_size]) * np.exp(0.1)
            
            z_sg=np.tile(z_sg,[1,FLAGS.visualize_length,1]).reshape(-1,FLAGS.sg_latent_size)
            
            rang= np.arange(-2,2,4/FLAGS.visualize_length)
            z_sg[fix_dim*FLAGS.visualize_length:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length,fix_dim]=rang
            #z_sg=z_sg[fix_dim*FLAGS.visualize_length+base:fix_dim*FLAGS.visualize_length+FLAGS.visualize_length+base]
   
            generated_adj, generated_adj_prob, spatial,  generated_node_feat= self.decoder(z_sg.astype('float32'))
            
            return generated_adj, generated_adj_prob, spatial,  generated_node_feat   
        
         
        
        
