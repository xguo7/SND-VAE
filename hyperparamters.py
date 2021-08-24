
#best parmters for synthetic 1 disentangled type
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
FLAGS.epochs=2000
FLAGS.dropout=1
FLAGS.batch_size=50
FLAGS.decoder_batch_size=50
FLAGS.num_feature=1
FLAGS.spatial_dim= 2
FLAGS.verbose=1
FLAGS.test_count=10


#best parmters for synthetic 2 disentangled type
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
FLAGS.sg_hidden_size=200
FLAGS.sg_latent_size=200
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
FLAGS.batch_size=50
FLAGS.decoder_batch_size=50
FLAGS.num_feature=1
FLAGS.spatial_dim= 2
FLAGS.verbose=1
FLAGS.test_count=10














