import os
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import math


tf.enable_eager_execution()
print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

from glob import glob
train_dataset_dir = glob("/hc/traditional compression/data/train2017/*")
print len(train_dataset_dir),'train images'

test_dataset_dir = glob("/hc/traditional compression/data/Kodak/*")
print len(test_dataset_dir),'Kodak images'

# IMAGE_SIZE = 256
batchsize = 1

def _load_data(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_image(image,channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.cond(tf.logical_and(tf.shape(image)[0]%64 != 0, tf.shape(image)[1]%64 != 0), lambda: tf.random_crop(image,[tf.shape(image)[0]-tf.shape(image)[0]%64,tf.shape(image)[1]-tf.shape(image)[1]%64,3],seed=1,name="cropper1"), lambda: tf.zeros([1,1]))
    # image = tf.cond(tf.logical_and(tf.shape(image)[0]%4 != 0, tf.shape(image)[1]%4 != 0), lambda: tf.image.resize_images(image,[256,256],method=2), lambda: tf.zeros([1,1]))
    # image = tf.cond(tf.logical_and(tf.shape(image)[0] > 1024, tf.shape(image)[1] > 1024), lambda: tf.random_crop(image,[256,256,3],seed=1,name="cropper2"), lambda: tf.zeros([1,1]))

    return image

# def _load_data_test(filename):
#     print filename
#     image = tf.read_file(filename)
#     image = tf.image.decode_image(image,channels=3)
#     image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#     # image = tf.cond(tf.logical_and(tf.shape(image)[0] > IMAGE_SIZE, tf.shape(image)[1] > IMAGE_SIZE), lambda: tf.random_crop(image,[IMAGE_SIZE,IMAGE_SIZE,3],seed=1,name="cropper"), lambda: tf.zeros([1,1]))

#     return image

def _skip_zeros(image):
    return tf.shape(image)[1] > 1

dataset = tf.data.Dataset.from_tensor_slices(train_dataset_dir)
dataset = dataset.map(_load_data,num_parallel_calls=1)
dataset = dataset.filter(_skip_zeros)
dataset = dataset.shuffle(buffer_size=1).batch(batchsize,drop_remainder=True).repeat(10) # 10 epoches, 32 batch-size


# test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset_dir)
# test_dataset = test_dataset.map(_load_data_test,num_parallel_calls=4)
# test_dataset = test_dataset.filter(_skip_zeros)
# test_dataset = test_dataset.shuffle(buffer_size=1).batch(1,drop_remainder=True).repeat(1) # 10 epoches, 32 batch-size  




# import sys
# sys.path.append("./compression-master")

# from layers import entropy_models
import network

N=192
M=24

#loss = tf.losses.mean_squared_error(next_element, rec) #(label, prediction)
lr = 1e-4
#optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
deepcoder = network.Deepcoder(M=M)
deepcoder_2 = network.Deepcoder(M=M)
deepcoder_4 = network.Deepcoder(M=M)
ms_discriminator = network.multiscale_discriminator() 

upscaling_2 = network.up()
upscaling_4 = network.up()

import entropy
# channels = 320
estimator = entropy.entropy_estimator(channels=M)
estimator_2 = entropy.entropy_estimator(channels=M)
estimator_4 = entropy.entropy_estimator(channels=M)

checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 deepcoder=deepcoder,
                                 deepcoder_2=deepcoder_2,
                                 deepcoder_4=deepcoder_4,
                                 estimator=estimator,
                                 estimator_2=estimator_2,
                                 estimator_4=estimator_4,
                                 ms_discriminator=ms_discriminator,
                                 upscaling_2=upscaling_2,
                                 upscaling_4=upscaling_4)

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# checkpoint.restore('./checkpoints/ckpt-59')
# print ('load latset model!')

import time
import numpy as np

# lambda_m = (255**2)*0.05
lambda_m = 12
# num_pixels = IMAGE_SIZE*IMAGE_SIZE*4.0

def train(dataset, epochs):  
  ls = 0.0
  bpp = 0.0   
  psnr = 0.0
  mode = 4
  bpp_1 = 0.0
  loss_G = 0.0
  loss_D = 0.0
  for epoch in range(epochs):
 
    start = time.time()
    for step, input_image in enumerate(dataset):
      # with tf.GradientTape() as model_tape:
      num_pixels = tf.cast((tf.shape(input_image)[1] * tf.shape(input_image)[2]),tf.float32)
      input_2 = tf.image.resize_images(input_image, [tf.shape(input_image)[1]/2,tf.shape(input_image)[2]/2],method=2)
      input_4 = tf.image.resize_images(input_image, [tf.shape(input_2)[1]/2,tf.shape(input_2)[2]/2],method=2)
      # input_2 = downscaling(input_image)
      real = input_image.numpy()
      real_2 = input_2.numpy()
      real_4 = input_4.numpy()
      # print 'original image: max value:',np.max(real),'min value:',np.min(real),'mean value:',np.mean(real)
      # print 'downscaled image: max value:',np.max(real_2),'min value:',np.min(real_2),'mean value:',np.mean(real_2)
      
      with tf.GradientTape() as D_tape:
        # print(input_image.shape)        
        
        #scale k/4
        # print input_4.shape
        encoded_4_raw, encoded_4, output_4 = deepcoder_4(input_4)
        encoded_4_x = tf.reshape(encoded_4,[-1,1,M])
        encoded_4_x = tf.transpose(encoded_4_x, perm=[2,1,0])
        likelihoods_4 = estimator_4(encoded_4_x)
        train_bpp_4 = tf.reduce_sum(tf.log(likelihoods_4)) / -tf.log(tf.constant(2.0))/num_pixels
        input_2_up = upscaling_4(output_4)
        # print output_4.shape

        #scale k/2
        input_2_res = input_2 - input_2_up
        encoded_2_raw, encoded_2, output_2_res = deepcoder_2(input_2_res)
        encoded_2_x = tf.reshape(encoded_2,[-1,1,M])
        encoded_2_x = tf.transpose(encoded_2_x, perm=[2,1,0])
        likelihoods_2 = estimator_2(encoded_2_x)
        train_bpp_2 = tf.reduce_sum(tf.log(likelihoods_2)) / -tf.log(tf.constant(2.0))/num_pixels
        output_2 = output_2_res + input_2_up
        input_up = upscaling_2(output_2)
        # print output_2.shape

        #scale k
        input_res = input_image - input_up
        encoded_raw, encoded, output_res = deepcoder_2(input_res)
        encoded_x = tf.reshape(encoded,[-1,1,M])
        encoded_x = tf.transpose(encoded_x, perm=[2,1,0])
        likelihoods = estimator(encoded_x)
        train_bpp = tf.reduce_sum(tf.log(likelihoods)) / -tf.log(tf.constant(2.0))/num_pixels
        output = output_res + input_up

        #discriminator
        D_x_512, D_x2_512, D_x4_512, Dk_x_512 = ms_discriminator(input_image)
        D_Gz_512, D_Gz2_512, D_Gz4_512, Dk_Gz_512 = ms_discriminator(output)

        #loss 
        D_loss_gan = tf.reduce_mean(tf.square(D_x_512 - 1.)) + tf.reduce_mean(tf.square(D_Gz_512))
        G_loss_gan = tf.reduce_mean(tf.square(D_Gz_512 - 1.))

        D_loss_gan_2 = tf.reduce_mean(tf.square(D_x2_512 - 1.)) + tf.reduce_mean(tf.square(D_Gz2_512))
        G_loss_gan_2 = tf.reduce_mean(tf.square(D_Gz2_512 - 1.))

        D_loss_gan_4 = tf.reduce_mean(tf.square(D_x4_512 - 1.)) + tf.reduce_mean(tf.square(D_Gz4_512))
        G_loss_gan_4 = tf.reduce_mean(tf.square(D_Gz4_512 - 1.))        

        D_x_layers, D_Gz_layers = [j for i in Dk_x_512 for j in i], [j for i in Dk_Gz_512 for j in i]
        feature_matching_loss = tf.reduce_sum([tf.reduce_mean(tf.abs(Dkx-Dkz)) for Dkx, Dkz in zip(D_x_layers, D_Gz_layers)])

        loss_mse_4 = tf.losses.mean_squared_error(input_4, output_4)
        loss_mse_2 = tf.losses.mean_squared_error(input_2, output_2)
        loss_mse = tf.losses.mean_squared_error(input_image, output)
        

        D_loss = D_loss_gan + D_loss_gan_2 + D_loss_gan_4
        mse_loss = lambda_m * loss_mse + 1e2 * lambda_m * loss_mse_2 + 1e2 * lambda_m * loss_mse_4
        bpp_loss = 10 * train_bpp_4 + 10 * train_bpp_2 + 1e3 * train_bpp
        G_loss = G_loss_gan + mse_loss + bpp_loss + 10 * feature_matching_loss 

        Variables_G = deepcoder_4.variables+estimator_4.variables+upscaling_4.variables+deepcoder_2.variables+estimator_2.variables+upscaling_2.variables+deepcoder.variables+estimator.variables    
        Variables_D = ms_discriminator.variables

        if mode == 4:
          # gradients_G = model_tape.gradient(G_loss, Variables_G)
          gradients_D = D_tape.gradient(D_loss, Variables_D)

        # if mode == 4:
        #   for i, g in enumerate(gradients):
        #       if g is not None:
        #           gradients[i] = (tf.clip_by_norm(g, 10))

        if mode == 4:
          optimizer.apply_gradients(zip(gradients_D, Variables_D))
          # optimizer.apply_gradients(zip(gradients_G, Variables_G))
        
      with tf.GradientTape() as G_tape:
        # print(input_image.shape)        
        
        #scale k/4
        encoded_4_raw, encoded_4, output_4 = deepcoder_4(input_4)
        encoded_4_x = tf.reshape(encoded_4,[-1,1,M])
        encoded_4_x = tf.transpose(encoded_4_x, perm=[2,1,0])
        likelihoods_4 = estimator_4(encoded_4_x)
        train_bpp_4 = tf.reduce_sum(tf.log(likelihoods_4)) / -tf.log(tf.constant(2.0))/num_pixels
        input_2_up = upscaling_4(output_4)

        #scale k/2
        input_2_res = input_2 - input_2_up
        encoded_2_raw, encoded_2, output_2_res = deepcoder_2(input_2_res)
        encoded_2_x = tf.reshape(encoded_2,[-1,1,M])
        encoded_2_x = tf.transpose(encoded_2_x, perm=[2,1,0])
        likelihoods_2 = estimator_2(encoded_2_x)
        train_bpp_2 = tf.reduce_sum(tf.log(likelihoods_2)) / -tf.log(tf.constant(2.0))/num_pixels
        output_2 = output_2_res + input_2_up
        input_up = upscaling_2(output_2)

        #scale k
        input_res = input_image - input_up
        encoded_raw, encoded, output_res = deepcoder_2(input_res)
        encoded_x = tf.reshape(encoded,[-1,1,M])
        encoded_x = tf.transpose(encoded_x, perm=[2,1,0])
        likelihoods = estimator(encoded_x)
        train_bpp = tf.reduce_sum(tf.log(likelihoods)) / -tf.log(tf.constant(2.0))/num_pixels
        output = output_res + input_up

        #discriminator
        D_x_512, D_x2_512, D_x4_512, Dk_x_512 = ms_discriminator(input_image)
        D_Gz_512, D_Gz2_512, D_Gz4_512, Dk_Gz_512 = ms_discriminator(output)

        #loss 
        D_loss_gan = tf.reduce_mean(tf.square(D_x_512 - 1.)) + tf.reduce_mean(tf.square(D_Gz_512))
        G_loss_gan = tf.reduce_mean(tf.square(D_Gz_512 - 1.))

        D_loss_gan_2 = tf.reduce_mean(tf.square(D_x2_512 - 1.)) + tf.reduce_mean(tf.square(D_Gz2_512))
        G_loss_gan_2 = tf.reduce_mean(tf.square(D_Gz2_512 - 1.))

        D_loss_gan_4 = tf.reduce_mean(tf.square(D_x4_512 - 1.)) + tf.reduce_mean(tf.square(D_Gz4_512))
        G_loss_gan_4 = tf.reduce_mean(tf.square(D_Gz4_512 - 1.))        

        D_x_layers, D_Gz_layers = [j for i in Dk_x_512 for j in i], [j for i in Dk_Gz_512 for j in i]
        feature_matching_loss = tf.reduce_sum([tf.reduce_mean(tf.abs(Dkx-Dkz)) for Dkx, Dkz in zip(D_x_layers, D_Gz_layers)])

        loss_mse_4 = tf.losses.mean_squared_error(input_4, output_4)
        loss_mse_2 = tf.losses.mean_squared_error(input_2, output_2)
        loss_mse = tf.losses.mean_squared_error(input_image, output)
        

        D_loss = D_loss_gan + D_loss_gan_2 + D_loss_gan_4
        mse_loss = lambda_m * loss_mse + 1e2 * lambda_m * loss_mse_2 + 1e2 * lambda_m * loss_mse_4
        bpp_loss = train_bpp_4 + train_bpp_2 + 1e2 * train_bpp
        G_loss = G_loss_gan + mse_loss + bpp_loss + 10 * feature_matching_loss 

        Variables_G = deepcoder_4.variables+estimator_4.variables+upscaling_4.variables+deepcoder_2.variables+estimator_2.variables+upscaling_2.variables+deepcoder.variables+estimator.variables    
        Variables_D = ms_discriminator.variables

        if mode == 4:
          gradients_G = G_tape.gradient(G_loss, Variables_G)
          # gradients_D = model_tape.gradient(D_loss, Variables_D)

        # if mode == 4:
        #   for i, g in enumerate(gradients):
        #       if g is not None:
        #           gradients[i] = (tf.clip_by_norm(g, 10))

        if mode == 4:
          # optimizer.apply_gradients(zip(gradients_D, Variables_D))
          optimizer.apply_gradients(zip(gradients_G, Variables_G))

      step_buf = np.minimum(1000.0,step+1)
      
      if mode == 4 or mode == 7:
        ls_current = loss_mse.numpy()

      ls = (ls*(step_buf-1) + ls_current)/step_buf
      if mode == 4 or mode == 7:
        bpp = (bpp*(step_buf-1) + (train_bpp+train_bpp_2+train_bpp_4).numpy())/step_buf
        bpp_1 = (bpp_1*(step_buf-1) + (train_bpp_2+train_bpp_4).numpy())/step_buf

      if mode == 4 or mode == 7:
        pr = 20 * math.log10(1 / math.sqrt(loss_mse.numpy()) + 1e-8)
        pr_1 = 20 * math.log10(1 / math.sqrt(loss_mse_2.numpy()) + 1e-8)
        loss_g = G_loss.numpy()
        loss_d = D_loss.numpy()
      loss_G = (loss_G*(step_buf-1) + loss_g)/step_buf
      loss_D = (loss_D*(step_buf-1) + loss_d)/step_buf
      psnr = (psnr*(step_buf-1) + pr)/step_buf

      if mode == 3 or mode == 4 or mode == 7:
        real = input_image[:1,:,:,:].numpy()
      if mode == 4 or mode == 7:
        recon = output[:1,:,:,:].numpy()

      # >>> Monitoring
      # tf.summary.scalar('learning_rate', learning_rate)
      # tf.summary.scalar('generator_loss', G_loss)
      # tf.summary.scalar('discriminator_loss', D_loss)
      # tf.summary.scalar('g_loss', G_loss_gan)
      # tf.summary.scalar('d_loss', D_loss_gan)
      # tf.summary.scalar('g_loss_2', G_loss_gan_2)
      # tf.summary.scalar('d_loss_2', D_loss_gan_2)
      # tf.summary.scalar('g_loss_4', G_loss_gan_4)
      # tf.summary.scalar('d_loss_4', D_loss_gan_4)
      # tf.summary.scalar('bpp', train_bpp)
      # tf.summary.scalar('bpp_2', train_bpp_2)
      # tf.summary.scalar('bpp_4', train_bpp_4)        
      # tf.summary.scalar('mse_loss', loss_mse)
      # tf.summary.scalar('mse_loss_2', loss_mse_2)
      # tf.summary.scalar('mse_loss_4', loss_mse_4)
      # tf.summary.scalar('feature_matching_loss', feature_matching_loss)
      # # tf.summary.image('real_images', input_image[:,:,:,:3], max_outputs=4)
      # # tf.summary.image('compressed_images', output[:,:,:,:3], max_outputs=4)
      # merge_op = tf.contrib.summary.merge_all()

      # train_writer = tf.contrib.summary.FileWriter(
      #     os.path.join('tensorboard', '{}_train_{}'.format(name, time.strftime('%d-%m_%I:%M'))), graph=tf.get_default_graph())

      import utils

      if (step + 1) % 1000 == 0:
        print 'step:',step+1,ls,ls_current,bpp,bpp_1
        print 'mode:',mode
        print ('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print ('Loss_G:{:.4f}'.format(loss_G))
        print ('Loss_D:{:.4f}'.format(loss_D))
        print ('psnr:{:.4f}'.format(psnr))
        print ('psnr_current:{:.4f}'.format(pr))
        print ('psnr from last scale:{:.4f}'.format(pr_1))
        utils.save_img(real,recon,step,epoch)
        start = time.time()
        # test.test(test_dataset,1)

      # if mode == 2 and psnr > 27:
      #   mode = 3
      # if mode == 3 and psnr > 26:
      #   mode = 4

      # import test
      # saving (checkpoint) the model every 1000 epochs
      if (step + 1) % 1000 == 0:
        # test.test(test_dataset,1)
        checkpoint.save(file_prefix = checkpoint_prefix)
        # train_writer.add_summary(merge_op)

#load checkpoint
# ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
# saver.restore(sess,ckpt.model_checkpoint_path)
train(dataset,100)
