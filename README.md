# Deep AutoEncoder-based Lossy Geometry Compression for Point Clouds
Created by Wei Yan


## Introduction
We propose a general autoencoder based architecture for lossy geometry point cloud compression.



## Dependencies
Requirements:
- Python 2.7+ with Numpy, Scipy and Matplotlib
- [Tensorflow (version 1.10+)](https://www.tensorflow.org/get_started/os_setup)
- [TFLearn](http://tflearn.org/installation)

Our code has been tested with Python 2.7, TensorFlow 1.10.0, TFLearn 0.3.2, CUDA 9.0 and cuDNN 7.0 on Ubuntu 14.04.


## Installation


To be able to train your own model you need first to _compile_ the EMD/Chamfer losses. In latent_3d_points_entropy/external/structural_losses we have included the cuda implementations of [Fan et. al](https://github.com/fanhqme/PointSetGeneration).
```
cd latent_3d_points_entropy/external

with your editor modify the first three lines of the makefile to point to 
your nvcc, cudalib and tensorflow library.

make
```

### Data Set
We provide ~57K point-clouds, each sampled from a mesh model of 
<a href="https://www.shapenet.org" target="_blank">ShapeNetCore</a> 
with (area) uniform sampling. To download them (1.4GB):
```
cd latent_3d_points_entropy/
./download_data.sh
```
The point-clouds will be stored in latent_3d_points_entropy/data/shape_net_core_uniform_samples_2048

Use the function snc_category_to_synth_id, defined in src/in_out/, to map a class name such as "chair" to its synthetic_id: "03001627". Point-clouds of models of the same class are stored under a commonly named folder.


### Usage
To train a point-cloud encoder and decoder look at:
    
    cd notebooks
    python train_single_class_ae.py
You can change the number of points and the category of training set at the top lines.


To compress a point cloud from shapenet:
    
    cd notebooks
    python compress.py --input_pointcloud_file \
     /YOUR/POINTCLOUD/FILE --output_dir /OUTPUT/DIR \
      --model_dir /PRETRAINED/MODEL/DIR  \ 
      --number_points point cloud number of farthest point sample. It must in accordance with model type!

For example:
    
    python compress.py --input_pointcloud_file /home/yw/Desktop/latent_3d_points_entropy/data/shape_net_core_uniform_samples_2048/03001627/1a6f615e8b1b5ae4dbbc9440457e303e.ply --output_dir ./compressed_data --model_dir /home/yw/Desktop/latent_3d_points_entropy/data/chair_model/chair_inout_point1024 --number_points 1024

 We provide four classes pretrained model (car,chair,table,airplane) in ./data.   

To decompress a point cloud from your latent code:

    cd notebooks 
    python decompress.py --latent_code_dir /YOUR/LATENT/CODE/DIR --decompress_dir /DECOMPRESS/DIR --model_dir /MODEL/DIR
 
 For example:
 
    python decompress.py --latent_code_dir /home/yw/Desktop/latent_3d_points_entropy/notebooks/compressed_data/recon_pc/1a6f615e8b1b5ae4dbbc9440457e303e.txt --decompress_dir ./decompressed_data --model_dir /home/yw/Desktop/latent_3d_points_entropy/data/chair_model/chair_inout_point1024

