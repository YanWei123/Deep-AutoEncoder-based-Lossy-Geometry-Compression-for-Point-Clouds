import os.path as osp
import sys
import os
import numpy as np
import six
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
from latent_3d_points_entropy.external.sampling.tf_sampling import farthest_point_sample,gather_point


import matplotlib.pylab as plt


top_out_dir = '../data/'          # Use to save Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

experiment_name = 'single_class_ae'
n_pc_points = 2048                # Number of points per model.
n_output_points = n_pc_points
bneck_size = 512                  # Bottleneck-AE size
ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer'
#class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
class_name = 'chair'




if class_name ==None:

    syn_id = snc_category_to_synth_id()
    all_syn_id = [k for v,k in six.iteritems(syn_id)]
    print(all_syn_id)
    print(len(all_syn_id))


    for i in range(len(all_syn_id)):
        id_i = all_syn_id[i]
        class_dir = osp.join(top_in_dir , id_i)
        if i==0:
            all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)
        else:
            all_pc_data.merge(load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True))     

    print(all_pc_data)
else:
    syn_id = snc_category_to_synth_id()[class_name]
    class_dir = osp.join(top_in_dir , syn_id)
    all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

# YW add: whether use FPS down sample

if n_pc_points<2048:
    new_pc = np.zeros([all_pc_data.num_examples,n_pc_points,3])

    print(new_pc.shape)
    
    with tf.Session('') as sess_new_pc:
        for i in range(all_pc_data.num_examples):
        #for i in range(1000):
        #raw_input()
            print(i)
            new_pc[i] =  sess_new_pc.run(gather_point(all_pc_data.point_clouds[i:i+1],\
                    farthest_point_sample(n_pc_points,all_pc_data.point_clouds[i:i+1])))

    print(new_pc[i].shape)
    del(all_pc_data.point_clouds)
    all_pc_data.point_clouds = new_pc
    print(all_pc_data.point_clouds.shape)


# split training and test dataset
seed = 100
all_pc_data.shuffle_data(seed)
num_example = all_pc_data.num_examples
test_set = PointCloudDataSet(all_pc_data.point_clouds[int(num_example*0.9):],init_shuffle=False)
all_pc_data = PointCloudDataSet(all_pc_data.point_clouds[:int(num_example*0.9)],init_shuffle=False)
print('training:',all_pc_data.num_examples)
print('test:',test_set.num_examples)

#class_dir = osp.join(top_in_dir , syn_id)
#all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

train_params = default_train_params()
train_params['batch_size'] = 8
train_params['z_rotate'] = False
train_params['training_epochs'] = 1200
print(train_params)

encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size,n_output_points=n_output_points)
train_dir = create_dir(osp.join(top_out_dir, experiment_name))

conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = train_params['z_rotate'],
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args,
            n_output = [n_output_points,3],
            training = True,
            lmbda = 4,
            lmbda2 = 1e-6
           )
conf.experiment_name = experiment_name
conf.held_out_step = 5   # How often to evaluate/print out loss on 
                         # held_out data (if they are provided in ae.train() ).
conf.save(osp.join(train_dir, 'configuration'))

# If you ran the above lines, you can reload a saved model like this:
load_pre_trained_ae = False
restore_epoch = 500
if load_pre_trained_ae:
    conf = Conf.load(train_dir + '/configuration')
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(conf.train_dir, epoch=restore_epoch)

# build AE model
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)

#Train the AE (save output to train_stats.txt) 
buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
train_stats = ae.train(all_pc_data, conf, log_file=fout)
fout.close()
