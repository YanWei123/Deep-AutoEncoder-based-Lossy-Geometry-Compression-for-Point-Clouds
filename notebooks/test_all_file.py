import os.path as osp
import os
import sys
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir,os.path.pardir))
sys.path.append(BASE_DIR)

from latent_3d_points_entropy.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points_entropy.src.autoencoder import Configuration as Conf
from latent_3d_points_entropy.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points_entropy.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder

from latent_3d_points_entropy.src.tf_utils import reset_tf_graph
from latent_3d_points_entropy.src.general_utils import plot_3d_point_cloud
from latent_3d_points_entropy.src.pc_util import write_ply
from latent_3d_points_entropy.external.sampling.tf_sampling import farthest_point_sample,gather_point


import matplotlib.pylab as plt




top_out_dir = '../data/'          # Use to save Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

experiment_name = 'single_class_ae'
n_pc_points = 2048                # Number of points per model.
bneck_size = 512                  # Bottleneck-AE size
ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer'
#class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
class_name = 'chair'

syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id)
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

# split training and test dataset
SPLIT = True
if SPLIT:

	seed = 100
	all_pc_data.shuffle_data(seed)
	num_example = all_pc_data.num_examples
	test_set = PointCloudDataSet(all_pc_data.point_clouds[int(num_example*0.9):],init_shuffle=False)
	all_pc_data = PointCloudDataSet(all_pc_data.point_clouds[:int(num_example*0.9)],init_shuffle=False)
	print('training:',all_pc_data.num_examples)
	print('test:',test_set.num_examples)

all_pc_data = test_set


if n_pc_points<2048:
	new_pc = np.zeros([all_pc_data.num_examples,n_pc_points,3])
	print(new_pc.shape)
	with tf.Session('') as sess_new_pc:
		for i in range(all_pc_data.num_examples):
			print(i)
			new_pc[i] =  sess_new_pc.run(gather_point(all_pc_data.point_clouds[i:i+1],\
                    farthest_point_sample(n_pc_points,all_pc_data.point_clouds[i:i+1])))

	print(new_pc[i].shape)
	all_pc_data_ori = PointCloudDataSet(all_pc_data.point_clouds,init_shuffle=False)
   	all_pc_data.point_clouds = new_pc
else:
	all_pc_data_ori = all_pc_data


# train_params = default_train_params()
# train_params['batch_size'] = 8
# print(train_params)

#encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
#train_dir = create_dir(osp.join(top_out_dir, experiment_name))
train_dir = osp.join(top_out_dir,experiment_name)


load_pre_trained_ae = True
restore_epoch = 1200
if load_pre_trained_ae:
    conf = Conf.load(train_dir + '/configuration')
    conf.training = False
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    #ae.restore_model(conf.train_dir, epoch=restore_epoch)
    ae.restore_model(train_dir, epoch=restore_epoch)

num_pc = all_pc_data.num_examples

re_pc = np.zeros([num_pc,n_pc_points,3])
latent_codes_all = []


RECON_DIR = os.path.join(top_out_dir,'recon_pc')
ORI_DIR = os.path.join(top_out_dir,'ori_pc')
BIN_DIR = os.path.join(top_out_dir,'recon_pc')
if not os.path.exists(RECON_DIR):
	os.makedirs(RECON_DIR)
if not os.path.exists(ORI_DIR):
	os.makedirs(ORI_DIR)
if not os.path.exists(BIN_DIR):
	os.makedirs(BIN_DIR)

SAVE_FILE = True

for i in range(num_pc):

	feed_pc, feed_model_names, _ = all_pc_data.next_batch(1)
	print('feed_pc.shape', feed_pc.shape)
	reconstructions = ae.reconstruct(feed_pc)[0]
	re_pc[i] = reconstructions

	print('reconstructions.shape',reconstructions.shape)
	latent_codes = ae.transform(feed_pc)
	print('latent_codes.shape',latent_codes.shape)
	print('latent_codes mean',np.mean(latent_codes))
	print('latent_codes:',latent_codes)
	latent_codes_all.append(latent_codes)

	if SAVE_FILE:

		write_ply(feed_pc[0],RECON_DIR+'/ori_%s_%d.ply'%(class_name,i))
		write_ply(all_pc_data_ori.point_clouds[i],ORI_DIR+'/ori_%s_%d.ply'%(class_name,i))
		write_ply(reconstructions[0],RECON_DIR+'/rec_%s_%d.ply'%(class_name,i))
		np.savetxt(RECON_DIR+'/%s_%d.txt'%(class_name,i),latent_codes[0])


# i = 1


# fig_re = plt.figure()
# a1 = fig_re.add_subplot(111,projection = '3d')
# a1.scatter(reconstructions[i][:, 0],reconstructions[i][:, 1], reconstructions[i][:, 2])
# plt.title('recon')

# fig_ori = plt.figure()
# a2 = fig_ori.add_subplot(111,projection = '3d')
# a2.scatter(feed_pc[i][:, 0], feed_pc[i][:, 1], feed_pc[i][:, 2])
# plt.title('origin')

# plt.show()




# if SAVE_FILE:

# 	RECON_DIR = os.path.join(top_out_dir,'recon_pc')
# 	BIN_DIR = os.path.join(top_out_dir,'recon_pc')
# 	for i in range(num_pc):
		


# i = 4
# plot_3d_point_cloud(reconstructions[i][:, 0], 
#                     reconstructions[i][:, 1], 
#                     reconstructions[i][:, 2], in_u_sphere=True);
