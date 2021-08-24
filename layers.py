from initializations import *
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs



class Graphite(Layer):
    """Graphite layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, batch_size,dropout=0., act=tf.nn.relu, **kwargs):
        super(Graphite, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.vars['weights']=tf.tile(tf.reshape(self.vars['weights'],[1,input_dim, output_dim]),[batch_size,1,1])    
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        x = inputs[0]
        recon_1 = inputs[1]
        recon_2 = inputs[2]
        x = tf.matmul(x, self.vars['weights'])
        x = tf.matmul(recon_1, tf.matmul(tf.transpose(recon_1,[0,2,1]), x)) + tf.matmul(recon_2, tf.matmul(tf.transpose(recon_2,[0,2,1]), x))
        outputs = self.act(x)
        return outputs


#class GraphConvolution(Layer):
#    """Basic graph convolution layer for undirected graph without edge labels."""
#    def __init__(self, input_dim, output_dim, batch_size,adj, dropout=0., act=tf.nn.relu, **kwargs):
#        super(GraphConvolution, self).__init__(**kwargs)
#        with tf.variable_scope(self.name + '_vars'):
#            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
#        self.vars['weights']=tf.tile(tf.reshape(self.vars['weights'],[1,input_dim, output_dim]),[batch_size,1,1])      
#        self.dropout = dropout
#        self.adj = adj
#        self.act = act
#
#    def _call(self, inputs):
#        x = inputs
#        x = tf.nn.dropout(x, self.dropout)
#        x = tf.matmul(x, self.vars['weights'])
#        x = tf.matmul(self.adj, x)
#        outputs = self.act(x)
#        return outputs
    
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)  
    
def GraphConvolution(adj, input_,output_dim, stddev=0.02,
           name="gcn"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        w_=tf.tile(tf.reshape(w,[1,input_.get_shape()[-1], output_dim]),[input_.get_shape()[0],1,1])      
        new_x=tf.matmul(input_, w_)
        conv = tf.matmul(adj,new_x)        
        outputs = lrelu(conv)
        
        return outputs
    
def GraphConvolution_full(adj, input_,output_dim, stddev=0.02,
           name="gcn"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        w_=tf.tile(tf.reshape(w,[1,input_.get_shape()[-1], output_dim]),[input_.get_shape()[0],1,1])      
        new_x=tf.tile(tf.reshape(tf.matmul(input_, w_),[FLAGS.batch_size,1,adj.get_shape()[1],-1]),[1,adj.get_shape()[-1],1,1])
        adj=tf.reshape(adj,[FLAGS.batch_size,adj.get_shape()[1],adj.get_shape()[1],-1])
        conv = tf.matmul(tf.transpose(adj,[0,3,1,2]),new_x)
        
        outputs = lrelu(conv)
        
        return tf.reshape(tf.transpose(outputs,[0,2,1,3]),[FLAGS.batch_size,adj.get_shape()[1],-1])
   


def SpatialGraphConvolution(adj, input_, rel, hidden_size, stddev=0.02, bias_start=0,
           name="sgconv"):
      with tf.variable_scope(name):
        
                num_batch=input_.get_shape()[0]
                num_node=input_.get_shape()[1]  
                input_dim=input_.get_shape()[2]
                rel_dim=rel.get_shape()[-1]
        
                rel_ij=tf.tile(tf.reshape(rel,[num_batch,num_node,num_node,1,-1]),[1,1,1,num_node,1])
                rel_jk=tf.tile(tf.reshape(rel,[num_batch,1,num_node,num_node,-1]),[1,num_node,1,1,1])
                dis_ik=tf.tile(tf.reshape(rel,[num_batch,num_node,1,num_node,-1]),[1,1,num_node,1,1])
                adj_3d=tf.multiply(tf.tile(tf.reshape(adj, [num_batch,num_node,num_node,1]),[1,1,1,num_node]), tf.tile(tf.reshape(adj, [-1,1,num_node,num_node]),[1,num_node,1,1]))

                
                matrix1 = tf.get_variable("Matrix1", [input_dim*3+rel_dim*2+1, hidden_size[0]], tf.float32,
                                   tf.random_normal_initializer(stddev=stddev))
                bias1 = tf.get_variable("bias1", [hidden_size[0]],        
                                   initializer=tf.constant_initializer(bias_start))
                matrix2 = tf.get_variable("Matrix2", [ input_dim*2+hidden_size[0]+rel_dim, hidden_size[1]], tf.float32,
                                   tf.random_normal_initializer(stddev=stddev))
                bias2 = tf.get_variable("bias2", [hidden_size[1]],        
                                   initializer=tf.constant_initializer(bias_start))                
                matrix3 = tf.get_variable("Matrix3", [ input_dim+hidden_size[1], hidden_size[2]], tf.float32,
                                   tf.random_normal_initializer(stddev=stddev))
                bias3 = tf.get_variable("bias3", [hidden_size[2]],        
                                   initializer=tf.constant_initializer(bias_start)) 
                
                node_feat_x=tf.tile(tf.reshape(input_,[num_batch,num_node,1,1,-1]),[1,1,num_node,num_node,1])
                node_feat_y=tf.tile(tf.reshape(input_,[num_batch,1,num_node,1,-1]),[1,num_node,1,num_node,1])
                node_feat_z=tf.tile(tf.reshape(input_,[num_batch,1,1,num_node,-1]),[1,num_node,num_node,1,1])                 
                m3=tf.concat([node_feat_x,node_feat_y,node_feat_z,rel_ij,rel_jk, dis_ik],axis=-1) #B*N*N*N*h'
                m3=tf.reshape(m3,[-1,input_dim*3+rel_dim*2+1])
                m3=tf.matmul(lrelu(m3), matrix1) + bias1 #B*N*N*N*h'' 
                m3=tf.reshape(m3,[num_batch,num_node,num_node,num_node,hidden_size[0]])                        
                m3=tf.transpose(m3,[0,1,2,4,3]) #B*N*N*h''*N
                # m3_sum=tf.matmul(m3, tf.reshape(adj_3d,[num_batch,num_node,num_node,num_node,1]))  #B*N*N*h''*1
                m3_sum=tf.matmul(m3, tf.reshape(adj_3d,[num_batch,num_node,num_node,num_node,1]))
                m3_sum=tf.reshape(m3_sum, [num_batch,num_node,num_node,-1]) #B*N*N*h''
                
                node_feat_x=tf.tile(tf.reshape(input_,[num_batch,num_node,1,-1]),[1,1,num_node,1])
                node_feat_y=tf.tile(tf.reshape(input_,[num_batch,1,num_node,-1]),[1,num_node,1,1])
                m2=tf.concat([node_feat_x,node_feat_y,tf.reshape(rel,[num_batch,num_node,num_node,-1]), m3_sum],axis=3) #B*N*N*h'          
                m2=tf.reshape(m2,[-1,input_dim*2+hidden_size[0]+rel_dim])
                m2=tf.matmul(lrelu(m2), matrix2) + bias2     
                m2=tf.reshape(m2,[num_batch,num_node,num_node,hidden_size[1]])     
                m2=tf.transpose(m2,[0,1,3,2])
                m2_sum=tf.matmul(m2, tf.reshape(adj,[num_batch,num_node,num_node,1]))  #B*N*h*1
                m2_sum=tf.reshape(m2_sum, [num_batch,num_node,-1]) #B*N*h
                
                m1=tf.concat([input_,m2_sum],axis=2)
                m1=tf.reshape(m1,[-1,input_dim+hidden_size[1]])
                m1=tf.matmul(lrelu(m1), matrix3) + bias3 
                output=tf.reshape(m1,[num_batch,num_node,-1])
                
                return output
        
def SpatialGraphConvolution_3D(adj, input_, rel, hidden_size, stddev=0.02, bias_start=0,
           name="sgconv"):
      with tf.variable_scope(name):
        
                num_batch=input_.get_shape()[0]
                num_node=input_.get_shape()[1]  
                input_dim=input_.get_shape()[2]
                rel_dim=rel.get_shape()[-1]
                #here the distence is the relations

                matrix0 = tf.get_variable("Matrix0", [input_dim*4+rel_dim*3+2, hidden_size[0]], tf.float32,
                                   tf.random_normal_initializer(stddev=stddev))
                bias0 = tf.get_variable("bias0", [hidden_size[0]],        
                                   initializer=tf.constant_initializer(bias_start))                
                matrix1 = tf.get_variable("Matrix1", [input_dim*3+rel_dim*2+hidden_size[0]+1, hidden_size[1]], tf.float32,
                                   tf.random_normal_initializer(stddev=stddev))
                bias1 = tf.get_variable("bias1", [hidden_size[1]],        
                                   initializer=tf.constant_initializer(bias_start))
                matrix2 = tf.get_variable("Matrix2", [ input_dim*2+rel_dim+hidden_size[1], hidden_size[2]], tf.float32,
                                   tf.random_normal_initializer(stddev=stddev))
                bias2 = tf.get_variable("bias2", [hidden_size[2]],        
                                   initializer=tf.constant_initializer(bias_start))                
                matrix3 = tf.get_variable("Matrix3", [ input_dim+hidden_size[2], hidden_size[3]], tf.float32,
                                   tf.random_normal_initializer(stddev=stddev))
                bias3 = tf.get_variable("bias3", [hidden_size[3]],        
                                   initializer=tf.constant_initializer(bias_start)) 

                node_feat_i=tf.tile(tf.reshape(input_,[num_batch,num_node,1,1,1,-1]),[1,1,num_node,num_node,num_node,1])
                node_feat_j=tf.tile(tf.reshape(input_,[num_batch,1,num_node,1,1,-1]),[1,num_node,1,num_node,num_node,1])
                node_feat_k=tf.tile(tf.reshape(input_,[num_batch,1,1,num_node,1,-1]),[1,num_node,num_node,1,num_node,1])                 
                node_feat_p=tf.tile(tf.reshape(input_,[num_batch,1,1,1,num_node,-1]),[1,num_node,num_node,num_node,1,1]) 
                rel_ij=tf.tile(tf.reshape(rel,[num_batch,num_node,num_node,1,1,-1]),[1,1,1,num_node,num_node,1])
                rel_jk=tf.tile(tf.reshape(rel,[num_batch,1,num_node,num_node,1,-1]),[1,num_node,1,1,num_node,1])
                rel_kp=tf.tile(tf.reshape(rel,[num_batch,1,1,num_node,num_node,-1]),[1,num_node,num_node,1,1,1])
                dis_ik=tf.tile(tf.reshape(rel,[num_batch,num_node,1,num_node,1,-1]),[1,1,num_node,1,num_node,1])                
                dis_ip=tf.tile(tf.reshape(rel,[num_batch,num_node,1,1,num_node,-1]),[1,1,num_node,num_node,1,1])  
                adj_4d=tf.multiply(tf.tile(tf.reshape(adj, [num_batch,num_node,num_node,1,1]),[1,1,1,num_node,num_node]), tf.tile(tf.reshape(adj, [num_batch,1,num_node,num_node,1]),[1,num_node,1,1,num_node]))                
                adj_4d=tf.multiply(adj_4d, tf.tile(tf.reshape(adj, [num_batch,1,1,num_node,num_node]),[1,num_node,num_node,1,1]))                
                m4=tf.concat([node_feat_i,node_feat_j,node_feat_k,node_feat_p,rel_ij,rel_jk, rel_kp, dis_ik, dis_ip],axis=-1) #B*N*N*N*N*h'
                m4=tf.reshape(m4,[-1,input_dim*4+rel_dim*3+2])
                m4=tf.matmul(lrelu(m4), matrix0) + bias0 #B*N*N*N*N*h''
                m4=tf.reshape(m4,[num_batch,num_node,num_node,num_node,num_node,hidden_size[0]])
                m4=tf.transpose(m4,[0,1,2,3,5,4]) #B*N*N*N*h''*N
                m4_sum=tf.matmul(m4, tf.reshape(adj_4d,[num_batch,num_node,num_node,num_node,num_node, 1]))  #B*N*N*N*h''*1
                m4_sum=tf.reshape(m4_sum, [num_batch,num_node,num_node,num_node,-1]) #B*N*N*N*h''
                
                node_feat_i=tf.tile(tf.reshape(input_,[num_batch,num_node,1,1,-1]),[1,1,num_node,num_node,1])
                node_feat_j=tf.tile(tf.reshape(input_,[num_batch,1,num_node,1,-1]),[1,num_node,1,num_node,1])
                node_feat_k=tf.tile(tf.reshape(input_,[num_batch,1,1,num_node,-1]),[1,num_node,num_node,1,1])  
                rel_ij=tf.tile(tf.reshape(rel,[num_batch,num_node,num_node,1,-1]),[1,1,1,num_node,1])
                rel_jk=tf.tile(tf.reshape(rel,[num_batch,1,num_node,num_node,-1]),[1,num_node,1,1,1])
                dis_ik=tf.tile(tf.reshape(rel,[num_batch,num_node,1,num_node,-1]),[1,1,num_node,1,1]) 
                adj_3d=tf.multiply(tf.tile(tf.reshape(adj, [num_batch,num_node,num_node,1]),[1,1,1,num_node]), tf.tile(tf.reshape(adj, [num_batch,1,num_node,num_node]),[1,num_node,1,1]))                
                m3=tf.concat([node_feat_i,node_feat_j,node_feat_k,rel_ij,rel_jk, dis_ik,m4_sum],axis=-1) #B*N*N*N*h'
                m3=tf.reshape(m3,[-1,input_dim*3+rel_dim*2++hidden_size[0]+1])
                m3=tf.matmul(lrelu(m3), matrix1) + bias1 #B*N*N*N*h''
                m3=tf.reshape(m3,[num_batch,num_node,num_node,num_node,hidden_size[1]])
                m3=tf.transpose(m3,[0,1,2,4,3]) #B*N*N*h''*N
                m3_sum=tf.matmul(m3, tf.reshape(adj_3d,[num_batch,num_node,num_node,num_node,1]))  #B*N*N*h''*1
                m3_sum=tf.reshape(m3_sum, [num_batch,num_node,num_node,-1]) #B*N*N*h''
                
                node_feat_i=tf.tile(tf.reshape(input_,[num_batch,num_node,1,-1]),[1,1,num_node,1])
                node_feat_j=tf.tile(tf.reshape(input_,[num_batch,1,num_node,-1]),[1,num_node,1,1])
                rel_ij=tf.reshape(rel,[num_batch,num_node,num_node,-1])
                m2=tf.concat([node_feat_i,node_feat_j,rel_ij, m3_sum],axis=3) #B*N*N*h'
                m2=tf.reshape(m2,[-1,input_dim*2+rel_dim+hidden_size[1]])
                m2=tf.matmul(lrelu(m2), matrix2) + bias2
                m2=tf.reshape(m2,[num_batch,num_node,num_node,hidden_size[2]])
                m2=tf.transpose(m2,[0,1,3,2])
                m2_sum=tf.matmul(m2, tf.reshape(adj,[num_batch,num_node,num_node,1]))  #B*N*h*1
                m2_sum=tf.reshape(m2_sum, [num_batch,num_node,-1]) #B*N*h

                m1=tf.concat([input_,m2_sum],axis=2)
                m1=tf.reshape(m1,[num_batch*num_node,-1])               
                m1=tf.matmul(lrelu(m1), matrix3) + bias3 
                output=tf.reshape(m1,[num_batch,num_node,-1])
                
                return output    
    
def SpatialGraphConvolution_3D_full(adj, input_, rel, hidden_size, stddev=0.02, bias_start=0,
           name="sgconv"):
    #this convolution if for the fully connected graph with edge weights
      with tf.variable_scope(name):
                
                num_batch=input_.get_shape()[0]
                num_node=input_.get_shape()[1]  
                input_dim=input_.get_shape()[2]   
                dis=rel
                rel_dim=rel.get_shape()[-1]+adj.get_shape()[-1]
                rel=tf.concat([rel,adj],axis=-1)
                
                #here the distence is the relations

                matrix0 = tf.get_variable("Matrix0", [input_dim*4+rel_dim*3+2, hidden_size[0]], tf.float32,
                                   tf.random_normal_initializer(stddev=stddev))
                bias0 = tf.get_variable("bias0", [hidden_size[0]],        
                                   initializer=tf.constant_initializer(bias_start))                
                matrix1 = tf.get_variable("Matrix1", [input_dim*3+rel_dim*2+hidden_size[0]+1, hidden_size[1]], tf.float32,
                                   tf.random_normal_initializer(stddev=stddev))
                bias1 = tf.get_variable("bias1", [hidden_size[1]],        
                                   initializer=tf.constant_initializer(bias_start))
                matrix2 = tf.get_variable("Matrix2", [ input_dim*2+rel_dim+hidden_size[1], hidden_size[2]], tf.float32,
                                   tf.random_normal_initializer(stddev=stddev))
                bias2 = tf.get_variable("bias2", [hidden_size[2]],        
                                   initializer=tf.constant_initializer(bias_start))                
                matrix3 = tf.get_variable("Matrix3", [ input_dim+hidden_size[2], hidden_size[3]], tf.float32,
                                   tf.random_normal_initializer(stddev=stddev))
                bias3 = tf.get_variable("bias3", [hidden_size[3]],        
                                   initializer=tf.constant_initializer(bias_start)) 

                node_feat_i=tf.tile(tf.reshape(input_,[num_batch,num_node,1,1,1,-1]),[1,1,num_node,num_node,num_node,1])
                node_feat_j=tf.tile(tf.reshape(input_,[num_batch,1,num_node,1,1,-1]),[1,num_node,1,num_node,num_node,1])
                node_feat_k=tf.tile(tf.reshape(input_,[num_batch,1,1,num_node,1,-1]),[1,num_node,num_node,1,num_node,1])                 
                node_feat_p=tf.tile(tf.reshape(input_,[num_batch,1,1,1,num_node,-1]),[1,num_node,num_node,num_node,1,1]) 
                rel_ij=tf.tile(tf.reshape(rel,[num_batch,num_node,num_node,1,1,-1]),[1,1,1,num_node,num_node,1])
                rel_jk=tf.tile(tf.reshape(rel,[num_batch,1,num_node,num_node,1,-1]),[1,num_node,1,1,num_node,1])
                rel_kp=tf.tile(tf.reshape(rel,[num_batch,1,1,num_node,num_node,-1]),[1,num_node,num_node,1,1,1])
                dis_ik=tf.tile(tf.reshape(dis,[num_batch,num_node,1,num_node,1,-1]),[1,1,num_node,1,num_node,1])                
                dis_ip=tf.tile(tf.reshape(dis,[num_batch,num_node,1,1,num_node,-1]),[1,1,num_node,num_node,1,1])  
                adj_4d=tf.ones([num_batch,num_node,num_node,num_node,num_node])
                m4=tf.concat([node_feat_i,node_feat_j,node_feat_k,node_feat_p,rel_ij,rel_jk, rel_kp, dis_ik, dis_ip],axis=-1) #B*N*N*N*N*h'
                m4=tf.reshape(m4,[-1,input_dim*4+rel_dim*3+2])
                m4=tf.matmul(lrelu(m4), matrix0) + bias0 #B*N*N*N*N*h''
                m4=tf.reshape(m4,[num_batch,num_node,num_node,num_node,num_node,hidden_size[0]])
                m4=tf.transpose(m4,[0,1,2,3,5,4]) #B*N*N*N*h''*N
                m4_sum=tf.matmul(m4, tf.reshape(adj_4d,[num_batch,num_node,num_node,num_node,num_node, 1]))  #B*N*N*N*h''*1
                m4_sum=tf.reshape(m4_sum, [num_batch,num_node,num_node,num_node,-1]) #B*N*N*N*h''
                
                node_feat_i=tf.tile(tf.reshape(input_,[num_batch,num_node,1,1,-1]),[1,1,num_node,num_node,1])
                node_feat_j=tf.tile(tf.reshape(input_,[num_batch,1,num_node,1,-1]),[1,num_node,1,num_node,1])
                node_feat_k=tf.tile(tf.reshape(input_,[num_batch,1,1,num_node,-1]),[1,num_node,num_node,1,1])  
                rel_ij=tf.tile(tf.reshape(rel,[num_batch,num_node,num_node,1,-1]),[1,1,1,num_node,1])
                rel_jk=tf.tile(tf.reshape(rel,[num_batch,1,num_node,num_node,-1]),[1,num_node,1,1,1])
                dis_ik=tf.tile(tf.reshape(dis,[num_batch,num_node,1,num_node,-1]),[1,1,num_node,1,1]) 
                adj_3d=tf.ones([num_batch,num_node,num_node,num_node])              
                m3=tf.concat([node_feat_i,node_feat_j,node_feat_k,rel_ij,rel_jk, dis_ik,m4_sum],axis=-1) #B*N*N*N*h'
                m3=tf.reshape(m3,[-1,input_dim*3+rel_dim*2++hidden_size[0]+1])
                m3=tf.matmul(lrelu(m3), matrix1) + bias1 #B*N*N*N*h''
                m3=tf.reshape(m3,[num_batch,num_node,num_node,num_node,hidden_size[1]])
                m3=tf.transpose(m3,[0,1,2,4,3]) #B*N*N*h''*N
                m3_sum=tf.matmul(m3, tf.reshape(adj_3d,[num_batch,num_node,num_node,num_node,1]))  #B*N*N*h''*1
                m3_sum=tf.reshape(m3_sum, [num_batch,num_node,num_node,-1]) #B*N*N*h''
                
                node_feat_i=tf.tile(tf.reshape(input_,[num_batch,num_node,1,-1]),[1,1,num_node,1])
                node_feat_j=tf.tile(tf.reshape(input_,[num_batch,1,num_node,-1]),[1,num_node,1,1])
                rel_ij=tf.reshape(rel,[num_batch,num_node,num_node,-1])
                m2=tf.concat([node_feat_i,node_feat_j,rel_ij, m3_sum],axis=3) #B*N*N*h'
                m2=tf.reshape(m2,[-1,input_dim*2+rel_dim+hidden_size[1]])
                m2=tf.matmul(lrelu(m2), matrix2) + bias2
                m2=tf.reshape(m2,[num_batch,num_node,num_node,hidden_size[2]])
                m2=tf.transpose(m2,[0,1,3,2])
                m2_sum=tf.matmul(m2, tf.reshape(tf.ones([num_batch,num_node,num_node]) ,[num_batch,num_node,num_node,1]))  #B*N*h*1
                m2_sum=tf.reshape(m2_sum, [num_batch,num_node,-1]) #B*N*h

                m1=tf.concat([input_,m2_sum],axis=2)
                m1=tf.reshape(m1,[num_batch*num_node,-1])               
                m1=tf.matmul(lrelu(m1), matrix3) + bias3 
                output=tf.reshape(m1,[num_batch,num_node,-1])
                
                return output    
            
            
class n2g(Layer):
    """aggregate node representation to graph level."""
    def __init__(self,input_dim, batch_size,dropout=0., act=tf.nn.relu, **kwargs):
        super(n2g, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim,20, name="weights")
        self.vars['weights']=tf.tile(tf.reshape(self.vars['weights'],[1,input_dim,20]),[batch_size,1,1])      
        self.dropout = dropout
        self.act = act
        self.diag=tf.tile(tf.reshape(tf.eye(input_dim),[1,input_dim,input_dim]),[batch_size,1,1])  
        

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, self.dropout)
        x = tf.multiply(tf.matmul(self.vars['weights'],x),self.diag)  #only left the diag values 
        outputs = self.act(x)
        return outputs
    
class g2n(Layer):
    """assgin graph representation to node level."""
    def __init__(self,input_dim, batch_size,dropout=0., act=tf.nn.relu, **kwargs):
        super(g2n, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(20,input_dim, name="weights")
        self.vars['weights']=tf.tile(tf.reshape(self.vars['weights'],[1,20,input_dim]),[batch_size,1,1])      
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, self.dropout)
        x = tf.matmul(self.vars['weights'],x)
        outputs = self.act(x)
        return outputs    



class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        x = tf.transpose(inputs,[0,2,1])
        x = tf.matmul(inputs, x)
        return x
    
def n2n(input_,output_dim,k_h, d_h=1, d_w=1, stddev=0.02,
           name="n2n"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [1, k_h, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())
        return conv 
    
def conv1d(input_,output_dim, filter_size, strides, name, stddev=0.02, padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [filter_size, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.layers.conv1d(input_, w, strides, padding=padding)
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())
        return conv      

def e2e(input_,output_dim,k_h, d_h=1, d_w=1, stddev=0.02,
           name="e2e"):
    with tf.variable_scope(name):
        w1 = tf.get_variable('w1', [1, k_h, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv1 = tf.nn.conv2d(input_, w1, strides=[1, d_h, d_w, 1], padding='SAME')
        biases1 = tf.get_variable('biases1', [output_dim], initializer=tf.constant_initializer(0.0))
        conv1 = tf.reshape(tf.nn.bias_add(conv1, biases1), conv1.get_shape())
        
        #w2 = tf.get_variable('w2', [k_h,k_h, input_.get_shape()[-1], output_dim],
         #                   initializer=tf.truncated_normal_initializer(stddev=stddev))
        #w2=w1
        conv2 = tf.nn.conv2d(input_, tf.transpose(w1,[1,0,2,3]), strides=[1, d_h, d_w, 1], padding='SAME')
        #biases2 = tf.get_variable('biases2', [output_dim], initializer=tf.constant_initializer(0.0))
        #biases2=biases1
        conv2 = tf.reshape(tf.nn.bias_add(conv2, biases1), conv2.get_shape())
        #m1 = tf.tile(conv1,[1,1,k_h,1])
        #m2 = tf.tile(conv2,[1,k_h,1,1])
        conv = tf.add(conv1, conv2)
        return conv
    
def e2n(input_,output_dim,k_h=50, d_h=1, d_w=1, stddev=0.02,
           name="e2n"):
     with tf.variable_scope(name):
        w = tf.get_variable('w', [1, k_h, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv
    
def n2g_adj(input_,output_dim,k_h, stddev=0.02,
           name="n2g"):
     with tf.variable_scope(name):
        w = tf.get_variable('w', [input_.get_shape()[1],1, 1, 1],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv,w

def de_n2g(input_, output_shape,
             k_h, d_h=1, d_w=1, stddev=0.02,
             name="de_n2g", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [input_.get_shape()[1],1, 1, 1],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w
        else:
            return deconv      

def de_e2n(input_, output_shape,
             k_h, d_h=1, d_w=1, stddev=0.02,
             name="de_e2n", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w1 = tf.get_variable('w1', [1,k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv1 = tf.nn.conv2d_transpose(input_, w1, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')
        biases1 = tf.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv1 = tf.reshape(tf.nn.bias_add(deconv1, biases1), deconv1.get_shape())
        
        #w2 = tf.get_variable('w2', [k_h,1,output_shape[-1], input_.get_shape()[-1]],
        #                    initializer=tf.random_normal_initializer(stddev=stddev))
        #w2=tf.transpose(w1,[1,0,2,3])
        deconv2 = tf.nn.conv2d_transpose(tf.transpose(input_,[0,2,1,3]), tf.transpose(w1,[1,0,2,3]), output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')
        #biases2 = tf.get_variable('biases2', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        #biases2=biases1
        deconv2 = tf.reshape(tf.nn.bias_add(deconv2, biases1), deconv1.get_shape())
        deconv=tf.add(deconv1,deconv2)
        
        if with_w:
            return deconv, w1
        else:
            return deconv 
        
def de_n2n(input_, output_shape,
             k_h, d_h=1, d_w=1, stddev=0.02,
             name="de_n2n", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [1,k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')
        biases = tf.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        
        if with_w:
            return deconv, w
        else:
            return deconv        
         
def de_e2e(input_, output_shape,
             k_h=50, d_h=1, d_w=1, stddev=0.02,
             name="de_e2e", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        input_1=tf.reshape(tf.reduce_sum(input_,axis=1),(int(input_.shape[0]),k_h,1,int(input_.shape[3]))) 
        input_2=tf.reshape(tf.reduce_sum(input_,axis=2),(int(input_.shape[0]),1,k_h,int(input_.shape[3]))) 
        
        w1 = tf.get_variable('w1', [1,k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv1 = tf.nn.conv2d_transpose(input_1, w1, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')       
        biases1 = tf.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv1 = tf.reshape(tf.nn.bias_add(deconv1, biases1), deconv1.get_shape())

        #w2 = tf.get_variable('w2', [k_h,1, output_shape[-1], input_.get_shape()[-1]],
         #                   initializer=tf.random_normal_initializer(stddev=stddev))
        #w2=tf.transpose(w1,[1,0,2,3])
        deconv2 = tf.nn.conv2d_transpose(input_2, tf.transpose(w1,[1,0,2,3]), output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')
        #biases2 = tf.get_variable('biases2', [output_shape[-1]], initializer=tf.constant_initializer(0.0))  
        #biases2=biases1
        deconv2 = tf.reshape(tf.nn.bias_add(deconv2, biases1), deconv2.get_shape())
        
        deconv=tf.add(deconv1,deconv2)/2
        if with_w:
            return deconv, w1
        else:
            return deconv  
        
def linear(input_, output_size, name, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
        
    

class batch_norm(object):
            # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)

def Graphite(inputs, output_dim, name, dropout=0., act=tf.nn.relu, stddev=0.02):
    """Graphite layer for undirected graph without edge labels."""
    with tf.variable_scope(name + '_vars'):
            shape = inputs[0].get_shape().as_list()
            matrix = tf.get_variable("Matrix", [shape[2], output_dim], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
    
            x = inputs[0]
            recon_1 = inputs[1]
            recon_2 = inputs[2]
            x = tf.matmul(x, matrix)
            x = tf.matmul(recon_1, tf.matmul(tf.transpose(recon_1,[0,2,1]), x)) + tf.matmul(recon_2, tf.matmul(tf.transpose(recon_2,[0,2,1]), x))
            outputs = act(x)
            return outputs        
        
def GeoGraphConvolution_adj_layer0(adj, input_, rel, output_dim, stddev=0.02,
                         name="geo_gcn"):
    with tf.variable_scope(name):
        adj = tf.multiply(adj, rel)
        w = tf.get_variable('w', [input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        w_=tf.tile(tf.reshape(w,[1,input_.get_shape()[-1], output_dim]),[input_.get_shape()[0],1,1])
        new_x=tf.tile(tf.reshape(tf.matmul(input_, w_),[FLAGS.batch_size,1,adj.get_shape()[1],-1]),[1,adj.get_shape()[-1],1,1])
        adj=tf.reshape(adj,[FLAGS.batch_size,adj.get_shape()[1],adj.get_shape()[1],-1])
        conv = tf.matmul(tf.transpose(adj,[0,3,1,2]),new_x)
        
        outputs = lrelu(conv)
                            
        return tf.reshape(tf.transpose(outputs,[0,2,1,3]),[FLAGS.batch_size,adj.get_shape()[1],-1])


def torch_gather(x, indices, gather_axis):
    # if pytorch gather indices are
    # [[[0, 10, 20], [0, 10, 20], [0, 10, 20]],
    #  [[0, 10, 20], [0, 10, 20], [0, 10, 20]]]
    # tf nd_gather needs to be
    # [[0,0,0], [0,0,10], [0,0,20], [0,1,0], [0,1,10], [0,1,20], [0,2,0], [0,2,10], [0,2,20],
    #  [1,0,0], [1,0,10], [1,0,20], [1,1,0], [1,1,10], [1,1,20], [1,2,0], [1,2,10], [1,2,20]]
    
    # create a tensor containing indices of each element
    all_indices = tf.where(tf.fill(indices.shape, True))
    gather_locations = tf.reshape(indices, [indices.shape.num_elements()])
    
    # splice in our pytorch style index at the correct axis
    gather_indices = []
    for axis in range(len(indices.shape)):
        if axis == gather_axis:
            gather_indices.append(tf.cast(gather_locations,tf.int64))
        else:
            gather_indices.append(all_indices[:, axis])

    gather_indices = tf.stack(gather_indices, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    reshaped = tf.reshape(gathered, indices.shape)
    return reshaped

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = tf.reshape(neighbor_idx,[neighbor_idx.shape[0], -1])
    neighbors_flat = tf.broadcast_to(tf.expand_dims(neighbors_flat,-1),[-1, -1, nodes.shape[2]])
    # Gather and re-pack
    neighbor_features = torch_gather(nodes, neighbors_flat, 1)
    neighbor_features = tf.reshape(neighbor_features, (list(neighbor_idx.shape)[:3] + [-1]))
    return neighbor_features

def quaternions(R):
    diag = tf.linalg.diag_part(R)
    Rxx, Ryy, Rzz = tf.unstack(diag, axis=-1)
    magnitudes = 0.5 * tf.sqrt(tf.abs(1 + tf.stack([
     Rxx - Ryy - Rzz,
     - Rxx + Ryy - Rzz,
     - Rxx - Ryy + Rzz
     ], -1)))
    _R = lambda i,j: R[:,:,:,i,j]
    signs = tf.sign(tf.stack([
     _R(2,1) - _R(1,2),
     _R(0,2) - _R(2,0),
     _R(1,0) - _R(0,1)
     ], axis=-1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = tf.sqrt(tf.nn.relu(1 + tf.reduce_sum(diag,-1, keepdims=True))) / 2.
    Q = tf.concat((xyz, w), -1)
    Q = tf.math.l2_normalize(Q, axis=-1)
    return Q

# X [B, L, 3]
def orientations(X, E_idx, eps=1e-6):
    # Pair features
    
    # Shifted slices of unit vectors
    dX = X[:,1:,:] - X[:,:-1,:]
    U = tf.math.l2_normalize(dX, axis=-1)
    u_2 = U[:,:-2,:]
    u_1 = U[:,1:-1,:]
    u_0 = U[:,2:,:]
    # Backbone normals
    n_2 = tf.math.l2_normalize(tf.linalg.cross(u_2, u_1), axis=-1)
    n_1 = tf.math.l2_normalize(tf.linalg.cross(u_1, u_0), axis=-1)
    
    # Bond angle calculation
    cosA = tf.reduce_sum(-(u_1 * u_0), axis=-1)
    cosA = tf.clip_by_value(cosA, -1+eps, 1-eps)
    A = tf.acos(cosA)
    # Angle between normals
    cosD = tf.reduce_sum((n_2 * n_1), -1)
    cosD = tf.clip_by_value(cosD, -1+eps, 1-eps)
    D = tf.sign(tf.reduce_sum((u_2 * n_1),-1)) * tf.acos(cosD)
    # Backbone features
    AD_features = tf.stack((tf.cos(A), tf.sin(A) * tf.cos(D), tf.sin(A) * tf.sin(D)), 2)
    AD_features = tf.pad(AD_features, [[0,0],[1,2],[0,0]])
    
    # Build relative orientations
    o_1 = tf.math.l2_normalize(u_2 - u_1, axis=-1)
    O = tf.stack((o_1, n_2, tf.linalg.cross(o_1, n_2)), 2)
    O = tf.reshape(O, list(O.shape[:2])+[9])
    O = tf.pad(O, [[0,0],[1,2],[0,0]])
    
    O_neighbors = gather_nodes(O, E_idx)
    X_neighbors = gather_nodes(X, E_idx)
    
    # Re-view as rotation matrices
    O = tf.reshape(O, list(O.shape[:2]) + [3,3])
    O_neighbors = tf.reshape(O_neighbors, list(O_neighbors.shape[:3]) + [3,3])
    
    # Rotate into local reference frames
    dX = X_neighbors - tf.expand_dims(X,-2)
    dU = tf.squeeze(tf.matmul(tf.expand_dims(O,2), tf.expand_dims(dX,-1)),-1)
    dU = tf.math.l2_normalize(dU, axis=-1)
    R = tf.matmul(tf.transpose(tf.expand_dims(O,2),[0,1,2,4,3]), O_neighbors)
    Q = quaternions(R)
    # Orientation features
    O_features = tf.concat((dU,Q), axis=-1)
    
    return AD_features, O_features

# D [B, L, k]
def rbf(D, num_rbf=16):
    D_min, D_max, D_count = 0., 20., num_rbf
    D_mu = tf.linspace(D_min, D_max, D_count)
    D_mu = tf.reshape(D_mu, [1, 1, 1, -1])
    D_sigma = (D_max-D_min)/D_count
    D_expand = tf.expand_dims(D, -1)
    RBF = tf.exp(-((D_expand-D_mu)/D_sigma)**2)
    return RBF

# X [B, L, 3]
def dist(X, eps=1e-6, top_k=10):
    dX = tf.expand_dims(X,1) - tf.expand_dims(X,2)
    D = tf.sqrt(tf.reduce_sum(dX**2, axis=3) + eps)
    D_max = tf.reduce_max(D, axis=-1, keepdims=True)
    D_adjust = D + D_max
    D_neighbors, E_idx = tf.math.top_k(-1.*D_adjust, top_k)
    return -1*D_neighbors, E_idx

def positionalEmbedding(edges, num_embeddings=16, period_range=[2,1000]):
    batch_size = edges.shape[0]
    node_size = edges.shape[1]
    neighbors = edges.shape[2]
    ii = tf.cast(tf.reshape(tf.range(node_size), [1, -1, 1]), tf.float32)
    d = tf.expand_dims((tf.cast(edges,tf.float32) - ii), -1)
    freq = tf.exp(tf.range(0, num_embeddings, 2, dtype=tf.float32) * -(np.log(10000.0)/num_embeddings))
    angles = d * tf.reshape(freq, [1, 1, 1, -1])
    E = tf.concat([tf.cos(angles), tf.sin(angles)], -1)
    return E


def StructGraphConvolution_adj_layer0(adj, input_, inputs_3d, output_dim, num_rbf=16, top_k=10, num_positional_embeddings=16, stddev=0.02, bias_start=0,
                               name="struct_gcn"):
    with tf.variable_scope(name):
        # build k-nearest neighbor graphs
        D_neighbors, E_idx = dist(inputs_3d)
        # pairwise features
        AD_features, O_features = orientations(inputs_3d, E_idx)
        RBF = rbf(D_neighbors)
        # pairwise embeddings
        E_positional = positionalEmbedding(E_idx)
        adj = tf.concat((E_positional,RBF,O_features),-1)
        print (adj.shape)
        matrix0 = tf.get_variable("edge_embedding_matrix", [num_positional_embeddings+num_rbf+7, 128], tf.float32,
                                  tf.random_normal_initializer(stddev=stddev))
        bias0 = tf.get_variable("bias1", [128], initializer=tf.constant_initializer(bias_start))
        adj = tf.matmul(adj, matrix0) + bias0
        w = tf.get_variable('w', [input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        w_=tf.tile(tf.reshape(w,[1,input_.get_shape()[-1], output_dim]),[input_.get_shape()[0],1,1])
        new_x=tf.tile(tf.reshape(tf.matmul(input_, w_),[FLAGS.batch_size,1,adj.get_shape()[1],-1]),[1,adj.get_shape()[-1],1,1])
        adj=tf.reshape(adj,[FLAGS.batch_size,adj.get_shape()[1],adj.get_shape()[1],-1])
        conv = tf.matmul(tf.transpose(adj,[0,3,1,2]),new_x)
        
        outputs = lrelu(conv)
                            
        return tf.reshape(tf.transpose(outputs,[0,2,1,3]),[FLAGS.batch_size,adj.get_shape()[1],-1])
        