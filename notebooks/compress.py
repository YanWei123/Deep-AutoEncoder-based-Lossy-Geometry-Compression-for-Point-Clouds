import os.path as osp
import os
import sys
import numpy as np
import argparse
import tensorflow as tf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir,os.path.pardir))
sys.path.append(BASE_DIR)

from latent_3d_points_entropy.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points_entropy.src.autoencoder import Configuration as Conf
from latent_3d_points_entropy.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points_entropy.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder, load_ply

from latent_3d_points_entropy.src.tf_utils import reset_tf_graph
from latent_3d_points_entropy.src.general_utils import plot_3d_point_cloud
from latent_3d_points_entropy.src.pc_util import write_ply
from latent_3d_points_entropy.external.sampling.tf_sampling import farthest_point_sample,gather_point


import matplotlib.pylab as plt

parser = argparse.ArgumentParser(description='Compression Argument')
parser.add_argument('--input_pointcloud_file',help='Input point cloud')
parser.add_argument('--output_dir',help='Output bitstream and reconstructed point cloud dir')
parser.add_argument('--model_dir',help='Pretrained model dir')
parser.add_argument('--number_points',type=int,help='Point cloud number of farthest point sample. It must in accordance with model type!')
args = parser.parse_args()

# input_pc_file = '/home/yw/Desktop/latent_3d_points_entropy/data/shape_net_core_uniform_samples_2048/03001627/1a6f615e8b1b5ae4dbbc9440457e303e.ply'
# top_out_dir = './compressed_data'
# model_dir= '/home/yw/Desktop/latent_3d_points_entropy/data/chair_model/single_class_ae_inout_point2048'
input_pc_file = args.input_pointcloud_file
top_out_dir = args.output_dir
model_dir = args.model_dir
n_pc_points = args.number_points

restore_epoch = 1200
# n_pc_points = 1024                # Number of points per model.


RECON_DIR = os.path.join(top_out_dir,'recon_pc')
ORI_DIR = os.path.join(top_out_dir,'ori_pc')
BIN_DIR = os.path.join(top_out_dir,'recon_pc')
if not os.path.exists(RECON_DIR):
	os.makedirs(RECON_DIR)
if not os.path.exists(ORI_DIR):
	os.makedirs(ORI_DIR)
if not os.path.exists(BIN_DIR):
	os.makedirs(BIN_DIR)

# load point cloud
print('Load point cloud from: ',input_pc_file)
pc = load_ply(input_pc_file)
print('Input point cloud shape:',pc.shape)
pc = np.expand_dims(pc,axis=0)

# If need downsample
if n_pc_points<2048:
	new_pc = np.zeros([n_pc_points,3])
	with tf.Session('') as sess_new_pc:
		new_pc = sess_new_pc.run(gather_point(pc,farthest_point_sample(n_pc_points,pc)))
	print('Downsampled point cloud shape: ',new_pc.shape)
	pc = new_pc




# load pretrained model and config
print('Load pretrained model from:', model_dir)
conf = Conf.load(os.path.join(model_dir,'configuration'))
conf.training = False
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)
#ae.restore_model(conf.train_dir, epoch=restore_epoch)
ae.restore_model(model_dir, epoch=restore_epoch)

# compress

reconstructions = ae.reconstruct(pc)[0][0]
print('reconstructed point cloud shape:',reconstructions.shape)

latent_codes = ae.transform(pc)
print('latent_codes.shape:',latent_codes.shape)

# save reconstructed pc and latent code
recon_file_name = input_pc_file.split('/')[-1].split('.')[0]
write_ply(reconstructions,os.path.join(RECON_DIR,recon_file_name+'_rec.ply'))
np.savetxt(os.path.join(RECON_DIR,recon_file_name+'.txt'),latent_codes[0])



# fig_re = plt.figure()
# a1 = fig_re.add_subplot(111,projection = '3d')
# a1.scatter(reconstructions[i][:, 0],reconstructions[i][:, 1], reconstructions[i][:, 2])
# plt.title('recon')

# fig_ori = plt.figure()
# a2 = fig_ori.add_subplot(111,projection = '3d')
# a2.scatter(feed_pc[i][:, 0], feed_pc[i][:, 1], feed_pc[i][:, 2])
# plt.title('origin')

# plt.show()


