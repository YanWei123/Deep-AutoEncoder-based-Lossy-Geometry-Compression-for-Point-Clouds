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



parser = argparse.ArgumentParser(description='Decompreesion Argument')
parser.add_argument('--latent_code_dir',help='Latent code dir')
parser.add_argument('--decompress_dir',help='Decompression dir')
parser.add_argument('--model_dir',help='Pretrained model dir')
args = parser.parse_args()

latent_code_dir = args.latent_code_dir
decompress_dir = args.decompress_dir
model_dir = args.model_dir

# latent_code_dir = '/home/yw/Desktop/latent_3d_points_entropy/notebooks/compressed_data/recon_pc/1a6f615e8b1b5ae4dbbc9440457e303e.txt'
# decompress_dir = './decompressed_data'
# model_dir= '/home/yw/Desktop/latent_3d_points_entropy/data/chair_model/single_class_ae_inout_point2048'


restore_epoch = 1200

if not os.path.exists(decompress_dir):
	os.makedirs(decompress_dir)

# load latent code
print('Load latent code from: ',latent_code_dir)
latent_code = np.loadtxt(latent_code_dir)



# load pretrained model and config
print('Load pretrained model from:', model_dir)
conf = Conf.load(os.path.join(model_dir,'configuration'))
conf.training = False
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)
#ae.restore_model(conf.train_dir, epoch=restore_epoch)
ae.restore_model(model_dir, epoch=restore_epoch)

# decompress
reconstructions = ae.decode(latent_code)[0]
print('reconstructed point cloud shape:',reconstructions.shape)

# compress
dec_file_name = latent_code_dir.split('/')[-1].split('.')[0]
write_ply(reconstructions,os.path.join(decompress_dir,dec_file_name+'_dec.ply'))




# fig_re = plt.figure()
# a1 = fig_re.add_subplot(111,projection = '3d')
# a1.scatter(reconstructions[:, 0],reconstructions[:, 1], reconstructions[:, 2])
# plt.title('recon')

# fig_ori = plt.figure()
# a2 = fig_ori.add_subplot(111,projection = '3d')
# a2.scatter(feed_pc[i][:, 0], feed_pc[i][:, 1], feed_pc[i][:, 2])
# plt.title('origin')

# plt.show()


