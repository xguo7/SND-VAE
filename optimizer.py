import tensorflow as tf
from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def DIP(enc_mean,lambda_od,lambda_d):
            # expectation of mu (mean of distributions)
            exp_mu = tf.reduce_mean(enc_mean, axis=0)
            # expectation of mu mu.tranpose
            mu_expand1 = tf.expand_dims(enc_mean, 1)
            mu_expand2 = tf.expand_dims(enc_mean, 2)
            exp_mu_mu_t = tf.reduce_mean( mu_expand1 * mu_expand2, axis=0)
            # covariance of model mean
            cov = exp_mu_mu_t - tf.expand_dims(exp_mu, 0) * tf.expand_dims(exp_mu, 1)
            diag_part = tf.diag_part(cov)
            off_diag_part = cov - tf.diag(diag_part)
            regulariser_od = lambda_od * tf.reduce_sum(off_diag_part**2)
            regulariser_d = lambda_d * tf.reduce_sum((diag_part - 1)**2)
            dip_vae_regulariser = regulariser_d + regulariser_od
            return dip_vae_regulariser
        
def gaussian_log_density(samples, mean, log_var):
  pi = tf.constant(np.math.pi)
  normalization = tf.log(2. * pi)
  inv_sigma = tf.exp(-log_var)
  tmp = (samples - mean) #[batch_size, batch_size, num_latents]
  return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)  
      
def total_correlation(z, z_mean, z_logstd):
  """Estimate of total correlation on a batch.
  We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
  log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
  for the minimization. The constant should be equal to (num_latents - 1) *
  log(batch_size * dataset_size)
  Args:
    z: [batch_size, num_latents]-tensor with sampled representation.
    z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
    z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
  Returns:
    Total correlation estimated on a batch.
  """
  z_logvar=tf.log(tf.multiply(tf.exp(z_logstd),tf.exp(z_logstd)))
  # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
  # tensor of size [batch_size, batch_size, num_latents]. In the following
  # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
  log_qz_prob = gaussian_log_density(
      tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
      tf.expand_dims(z_logvar, 0))
  # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
  # + constant) for each sample in the batch, which is a vector of size
  # [batch_size,].
  log_qz_product = tf.reduce_sum(
      tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
      axis=1,
      keepdims=False)
  # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
  # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
  log_qz = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  return tf.reduce_mean(log_qz - log_qz_product) 

def hierarchical_total_correlation(z1, z1_mean, z1_logstd,z2, z2_mean, z2_logstd,z3, z3_mean, z3_logstd):
  """Estimate of total correlation on a batch.
  We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
  log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
  for the minimization. The constant should be equal to (num_latents - 1) *
  log(batch_size * dataset_size)
  Args:
    z: [batch_size, num_latents]-tensor with sampled representation.
    z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
    z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
  Returns:
    Total correlation estimated on a batch.
  """
  # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
  # tensor of size [batch_size, batch_size, num_latents]. In the following
  # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
  z1_logvar=tf.log(tf.multiply(tf.exp(z1_logstd),tf.exp(z1_logstd)))
  z2_logvar=tf.log(tf.multiply(tf.exp(z2_logstd),tf.exp(z2_logstd)))
  z3_logvar=tf.log(tf.multiply(tf.exp(z3_logstd),tf.exp(z3_logstd)))
  z=tf.concat((z1,z2,z3),axis=1)
  dim1=z1.shape[1]
  dim2=z2.shape[1]+dim1
  dim3=z3.shape[1]+dim2
  z_mean=tf.concat((z1_mean,z2_mean,z3_mean),axis=1)
  z_logvar=tf.concat((z1_logvar,z2_logvar,z3_logvar),axis=1)
  
  log_qz_prob = gaussian_log_density(
      tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
      tf.expand_dims(z_logvar, 0))
  # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
  # + constant) for each sample in the batch, which is a vector of size
  # [batch_size,].
  log_qz1 = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob[:,:,0:dim1], axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  log_qz2 = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob[:,:,dim1:dim2], axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  log_qz3 = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob[:,:,dim2:dim3], axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  log_qz_product = log_qz1+log_qz2+log_qz3
  # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
  # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
  log_qz = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  return tf.reduce_mean(log_qz - log_qz_product)    
    
        
def KL_div2(mu, sigma, mu1, sigma1):
    '''KL divergence between N(mu,sigma**2) and N(mu1,sigma1**2)'''
    return 0.5 * ((sigma/sigma1)**2 + (mu - mu1)**2/sigma1**2 - 1 + 2*(tf.log(sigma1) - tf.log(sigma)))        
        
class OptimizerVAE(object):
    def __init__(self, preds_edge, preds_node, preds_spatial, labels_edge,labels_node, labels_spatial, labels_rel, model, num_nodes, pos_weight, norm,beta, global_iter):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)                        
        preds_edge = tf.reshape(preds_edge, [FLAGS.decoder_batch_size,num_nodes, num_nodes, -1])
        # preds_spatial = tf.reshape(preds_spatial, [FLAGS.decoder_batch_size,FLAGS.sampling_num, num_nodes, FLAGS.spatial_dim])
        # preds_node = tf.reshape(preds_node, [FLAGS.decoder_batch_size,FLAGS.sampling_num, num_nodes, FLAGS.num_feature])
        # preds_node = tf.reduce_mean(preds_node, axis=1)
        # preds_spatial = tf.reduce_mean(preds_spatial, axis=1)
        # preds_edge = tf.reduce_mean(preds_edge, axis=1)
        # preds_edge = tf.clip_by_value(preds_edge*2, 0., 1.)
        # preds_edge = preds_edge/tf.linalg.norm(preds_edge, axis=-1, keepdims=True)
        # preds_edge = tf.where(preds_edge >= FLAGS.edge_threshold, tf.ones_like(preds_edge), tf.zeros_like(preds_edge))
        labels_edge=tf.reshape(labels_edge,[FLAGS.decoder_batch_size, num_nodes, num_nodes, 1])
        #labels_edge= tf.concat([1-labels_edge, labels_edge],axis=-1) 
        
        if FLAGS.dataset == 'scene':
            labels_edge=tf.one_hot(indices=tf.cast(labels_edge, tf.int32), depth=FLAGS.num_edge_feature)
            labels_edge=tf.squeeze(labels_edge)
        else:
            labels_edge= tf.concat([1-labels_edge, labels_edge],axis=-1) 
           
        self.adj_cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_edge, logits=preds_edge))
        
        if FLAGS.dataset == 'scene':
            self.node_cost=0 #tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_node, logits=model.generated_node_feat_prob))
        else:
            self.node_cost=tf.reduce_mean(tf.math.squared_difference(labels_node,preds_node))
      

        #if FLAGS.model_type in ['disentangled','disentangled_C','NED-VAE-IP','beta-TCVAE']:
        self.spatial_cost=tf.reduce_mean(tf.math.squared_difference(labels_spatial,preds_spatial))
        #else:
        #self.spatial_cost=tf.reduce_mean(tf.math.squared_difference(labels_rel,preds_spatial))
           
        self.mse_loss=self.adj_cost+self.node_cost+self.spatial_cost
         
        if FLAGS.model_type in ['disentangled','geoGCN','posGCN']:
              self.kl_s = -(0.5) * tf.reduce_mean(1 + 2 * model.z_std_s - tf.square(model.z_mean_s) - tf.square(tf.exp(model.z_std_s))) #actually kl
              self.kl_g = -(0.5) * tf.reduce_mean(1 + 2 * model.z_std_g - tf.square(model.z_mean_g) - tf.square(tf.exp(model.z_std_g)))
              self.kl_sg = -(0.5) * tf.reduce_mean(1 + 2 * model.z_std_sg - tf.square(model.z_mean_sg) - tf.square(tf.exp(model.z_std_sg)))                           
              self.kl=self.kl_sg + self.kl_s+ self.kl_g 
              self.cost =self.mse_loss+beta*self.kl

        elif FLAGS.model_type == 'disentangled_C':
              self.kl_s = -(0.5) * tf.reduce_mean(1 + 2 * model.z_std_s - tf.square(model.z_mean_s) - tf.square(tf.exp(model.z_std_s))) #actually kl
              self.kl_g = -(0.5) * tf.reduce_mean(1 + 2 * model.z_std_g - tf.square(model.z_mean_g) - tf.square(tf.exp(model.z_std_g)))
              self.kl_sg = -(0.5) * tf.reduce_mean(1 + 2 * model.z_std_sg - tf.square(model.z_mean_sg) - tf.square(tf.exp(model.z_std_sg))) 
              # C = tf.clip_by_value(FLAGS.C_max/FLAGS.C_stop_iter*global_iter, 0, FLAGS.C_max)
              C = tf.clip_by_value(FLAGS.C_max*FLAGS.C_step/FLAGS.C_stop_iter*(global_iter//FLAGS.C_step), 0, FLAGS.C_max)
              self.kl=FLAGS.gamma*tf.nn.relu(self.kl_sg-C) + self.kl_s+ self.kl_g                           
            #   self.kl=self.kl_sg + self.kl_s+ self.kl_g
              self.cost = self.mse_loss + self.kl

        elif FLAGS.model_type == 'NED-VAE-IP':
            self.kl_s = -(0.5) * tf.reduce_mean(1 + 2 * model.z_std_s - tf.square(model.z_mean_s) - tf.square(tf.exp(model.z_std_s))) #actually -kl
            self.kl_g = -(0.5) * tf.reduce_mean(1 + 2 * model.z_std_g - tf.square(model.z_mean_g) - tf.square(tf.exp(model.z_std_g)))
            self.kl_sg = -(0.5) * tf.reduce_mean(1 + 2 * model.z_std_sg - tf.square(model.z_mean_sg) - tf.square(tf.exp(model.z_std_sg))) 
            self.kl=self.kl_sg + self.kl_s+ self.kl_g
            dip_regulizar=DIP(model.z_mean_s,10,100)+DIP(model.z_mean_g,10,100)+DIP(model.z_mean_sg,10,100)
            self.cost=self.mse_loss+self.kl+beta*dip_regulizar

        elif FLAGS.model_type == 'beta-TCVAE':
                self.kl_s = -(0.5) * tf.reduce_mean(1 + 2 * model.z_std_s - tf.square(model.z_mean_s) - tf.square(tf.exp(model.z_std_s))) #actually -kl
                self.kl_g = -(0.5) * tf.reduce_mean(1 + 2 * model.z_std_g - tf.square(model.z_mean_g) - tf.square(tf.exp(model.z_std_g)))
                self.kl_sg = -(0.5) * tf.reduce_mean(1 + 2 * model.z_std_sg - tf.square(model.z_mean_sg) - tf.square(tf.exp(model.z_std_sg))) 
                self.kl=self.kl_sg + self.kl_s+ self.kl_g  
                self.cost =self.mse_loss+beta*self.kl
                self.cost+=10*(total_correlation(model.z_s, model.z_mean_s, model.z_std_s)+total_correlation(model.z_g,model.z_mean_g, model.z_std_g)+total_correlation(model.z_sg,model.z_mean_sg, model.z_std_sg))
 
        else:
              self.kl_sg=-(0.5) * tf.reduce_mean(1 + 2 * model.z_std_sg - tf.square(model.z_mean_sg) - tf.square(tf.exp(model.z_std_sg)))
              self.cost =self.mse_loss+beta*self.kl_sg

        #self.cost+=0.003*l2_norm 
        self.opt_op=self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
       
        if FLAGS.model_type in ['disentangled', 'disentangled_C','NED-VAE-IP','beta-TCVAE','geoGCN','posGCN']:
              self.overall_loss=[self.cost, self.spatial_cost, self.adj_cost, self.node_cost, self.kl_g, self.kl_s, self.kl_sg]
        else:
              self.overall_loss=[self.cost, self.spatial_cost, self.adj_cost, self.node_cost, self.kl_sg]
            



