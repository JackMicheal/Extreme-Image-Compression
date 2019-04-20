import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import math


tf.enable_eager_execution()
print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

from glob import glob
train_dataset_dir = glob("/hc/traditional compression/data/Kodak/*")
print len(train_dataset_dir),'Kodak images'

# IMAGE_SIZE = 512

def _load_data(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_image(image,channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # image = tf.cond(tf.logical_and(tf.shape(image)[0] > IMAGE_SIZE, tf.shape(image)[1] > IMAGE_SIZE), lambda: tf.random_crop(image,[IMAGE_SIZE,IMAGE_SIZE,3],seed=1,name="cropper"), lambda: tf.zeros([1,1]))

    return image

def _skip_zeros(image):
    return tf.shape(image)[1] > 1

def cdf_hyper(x,loc,scale):
  mask_r = tf.math.greater(x,loc)
  mask_l = tf.math.less_equal(x,loc)
  c_l = 1.0/2.0 * tf.exp(-tf.abs(x - loc)/scale)
  c_r = 1.0 - 1.0/2.0 * tf.exp(-tf.abs(x - loc)/scale)
  c = c_l*tf.cast(mask_l,dtype=tf.float32) + c_r*tf.cast(mask_r,dtype=tf.float32)
  return c

dataset = tf.data.Dataset.from_tensor_slices(train_dataset_dir)
dataset = dataset.map(_load_data,num_parallel_calls=4)
dataset = dataset.filter(_skip_zeros)
dataset = dataset.shuffle(buffer_size=1).batch(1,drop_remainder=True).repeat(1) # 10 epoches, 32 batch-size

import h5py


# import sys
# sys.path.append("./compression-master")

# from layers import entropy_models
import network

N=192
M=320
M1=320

#loss = tf.losses.mean_squared_error(next_element, rec) #(label, prediction)
lr = 1e-4
#optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
deepcoder = network.Deepcoder_Relu(M=M)
deepcoder_2 = network.Deepcoder_Relu_Up(M=M1)
# deepcoder_4 = network.Deepcoder(M=M1)
# downscaling = network.downscaling()
# inception_model = network.inception_model()

# upscaling_2 = network.upscaling()
# upscaling_4 = network.upscaling()

import entropy
# channels = 320
#entropy_bottleneck = entropy_models.EntropyBottleneck()
# estimator = entropy.entropy_estimator(channels=M)
estimator_2 = entropy.entropy_estimator(channels=M1)
# estimator_4 = entropy.entropy_estimator(channels=M1)

checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 # downscaling=downscaling,
                                 deepcoder=deepcoder,
                                 deepcoder_2=deepcoder_2,
                                 # inception_model=inception_model,
                                 # estimator=estimator,
                                 estimator_2=estimator_2)


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# checkpoint.restore('./checkpoints/ckpt-57')
print ('load latset model!')

import time
import numpy as np


def test(dataset, epochs):  
  ls = 0.0
  bpp = 0.0   
  psnr = 0.0
  psnr_2 = 0.0
  bpp_2 = 0.0
  mode = 4
  for epoch in range(epochs):
 
    start = time.time()
    for step, input_image in enumerate(dataset):
      with tf.GradientTape() as model_tape:
        # print(step)
        # print(input_image.shape)
        num_pixels = tf.cast((tf.shape(input_image)[1]*tf.shape(input_image)[2]),tf.float32)
        input_2 = tf.image.resize_images(input_image, [tf.shape(input_image)[1]/4,tf.shape(input_image)[2]/4],method=2)
        real = input_image.numpy()
        real_2 = input_2.numpy()
        print 'original image: max value:',np.max(real),'min value:',np.min(real),'mean value:',np.mean(real)
        print 'downscaled image: max value:',np.max(real_2),'min value:',np.min(real_2),'mean value:',np.mean(real_2)

        #scale k/2
        encoded_2, output_2, loc, scale = deepcoder_2(input_2)
        encoded_2_x = tf.reshape(encoded_2,[-1,1,M])
        encoded_2_x = tf.transpose(encoded_2_x, perm=[2,1,0])
        likelihoods_2 = estimator_2(encoded_2_x)
        train_bpp_2 = tf.reduce_sum(tf.log(likelihoods_2)) / -tf.log(tf.constant(2.0))/num_pixels
        # output_2 = tf.nn.sigmoid(output_2)

        # #upscaling k/2
        # input_up = upscaling_2(output_2)

        #scale k
        input_res = input_image - output_2
        encoded_raw, encoded, output_res = deepcoder(input_res)

        upper = encoded + 0.5
        lower = encoded - 0.5
        sign = tf.math.sign(upper + lower - loc)

        upper = - sign * (upper - loc) + loc
        lower = - sign * (lower - loc) + loc

        upper = cdf_hyper(upper,loc,scale)
        lower = cdf_hyper(lower,loc,scale)

        p_laplace = tf.math.abs(upper - lower)
        p_laplace = tf.maximum(p_laplace,1e-9)

        train_bpp = tf.reduce_sum(tf.log(p_laplace)) / -tf.log(tf.constant(2.0))/num_pixels
        output = output_2 + output_res
        # output = inception_model(output)


        # output_save = tf.nn.sigmoid(output)
        ones = tf.ones_like(output)
        zeros = tf.zeros_like(output)
        output = tf.where(tf.greater(output,ones),ones,output)
        output = tf.where(tf.greater(zeros,output),zeros,output)
        loss_mse_2 = tf.losses.mean_squared_error(input_image, output_2)
        loss_mse = tf.losses.mean_squared_error(input_image, output)
        
        # loss = lambda_m * loss_mse + train_bpp + lambda_m * loss_mse_2 + 30 * train_bpp_2
        # loss_2 = lambda_m * loss_mse_2 + 30 * train_bpp_2

        # loss_up_2 = lambda_m * tf.losses.mean_squared_error(input_image, input_up) + lambda_m * loss_mse_2 + 50 * train_bpp_2

        # loss_tune_2 = loss_2 + loss_up_2 + loss
        # loss_all = loss_2 + loss_up_2 + 10 * loss

        
        # for v in upscaling_2.variables:
        #   Variables_2_up.append(v)
        #   Variables.append(v)

        # Variables_tune_2 = deepcoder_2.variables+estimator_2.variables+upscaling_2.variables+deepcoder.variables+estimator.variables    
        # Variables_all = deepcoder_2.variables+estimator_2.variables+upscaling_2.variables+deepcoder.variables+estimator.variables 

        

        # optimizer.minimize(loss)
        
        step_buf = np.minimum(1000.0,step+1)
        
        if mode == 4 or mode == 7:
          ls_current = loss_mse.numpy()
        if mode == 2: 
          ls_current = loss_mse_2.numpy()
        if mode == 0:
          ls_current = loss_mse_4.numpy()
        if mode == 3:
          ls_current = loss_up_2.numpy()
        if mode == 1:
          ls_current = loss_up_4.numpy()

        ls = (ls*(step_buf-1) + ls_current)/step_buf
        if mode == 4 or mode == 7:
          bpp = (bpp*(step_buf-1) + (train_bpp+train_bpp_2).numpy())/step_buf
          bpp_2 = (bpp_2*(step_buf-1) + (train_bpp_2).numpy())/step_buf
          bpp_current = (train_bpp+train_bpp_2).numpy()
          bpp_2_current = (train_bpp_2).numpy()
        if mode == 2 or mode == 3:
          bpp = (bpp*(step_buf-1) + (train_bpp_2).numpy())/step_buf
        if mode == 0 or mode == 1:
          bpp = (bpp*(step_buf-1) + train_bpp_4.numpy())/step_buf

        if mode == 4 or mode == 7:
          pr = 20 * math.log10(1 / math.sqrt(loss_mse.numpy()) + 1e-8)
          pr_1 = 20 * math.log10(1 / math.sqrt(loss_mse_2.numpy()) + 1e-8)
        if mode == 2:
          pr = 20 * math.log10(1 / math.sqrt(loss_mse_2.numpy()) + 1e-8)
        if mode == 0:
          pr = 20 * math.log10(1 / math.sqrt(loss_mse_4.numpy()) + 1e-8)
        if mode == 3:
          pr = 20 * math.log10(1 / math.sqrt(loss_up_2.numpy()) + 1e-8)
        if mode == 1:
          pr = 20 * math.log10(1 / math.sqrt(loss_up_4.numpy()) + 1e-8)
        psnr = (psnr*(step_buf-1) + pr)/step_buf
        psnr_2 = (psnr_2*(step_buf-1) + pr_1)/step_buf

        if mode == 3 or mode == 4 or mode == 7:
          real = input_image[:1,:,:,:].numpy()
          real_1 = input_res[:1,:,:,:].numpy()
        if mode == 1 or mode == 2:
          real = input_2[:1,:,:,:].numpy()
        if mode == 0:
          real = input_4[:1,:,:,:].numpy()
        if mode == 4 or mode == 7:
          recon = output[:1,:,:,:].numpy()
          recon_1 = output_2[:1,:,:,:].numpy()
          res = output_res[:1,:,:,:].numpy()
        if mode == 2:
          recon = output_2[:1,:,:,:].numpy()
        if mode == 0:
          recon = output_4[:1,:,:,:].numpy()
        if mode == 1:
          recon = input_2_up[:1,:,:,:].numpy()
        if mode == 3:
          recon = input_up[:1,:,:,:].numpy()

      import utils

      if (step + 1) % 1 == 0:
        print 'step:',step+1,ls,ls_current,bpp,bpp_2,bpp_current,bpp_2_current
        print 'mode:',mode
        print ('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        # print ('bpp:{:.4f}'.format(bpp))
        print ('psnr:{:.4f}'.format(psnr))
        print ('psnr of scale k/2:{:.4f}'.format(psnr_2))
        print ('psnr_current:{:.4f}'.format(pr))
        print ('psnr from last scale:{:.4f}'.format(pr_1))
        utils.save_img_test(real,recon,step,epoch)
        utils.save_img_res(real_1,res,recon_1,step,epoch)
        start = time.time()



#load checkpoints
test(dataset,1)
