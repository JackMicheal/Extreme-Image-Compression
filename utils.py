from PIL import Image
import tensorflow as tf
import numpy as np

def save_img(img1,img2,step,epoch):
	img1 = np.squeeze(img1)
	img2 = np.squeeze(img2)
	print(img1.shape)
	img1[img1>1.]=1.
	img1[img1<0.]=0.
	img2[img2>1.]=1.
	img2[img2<0.]=0.
	img1 = img1*255
	img2 = img2*255
	img1 = img1.astype(np.uint8)
	img2 = img2.astype(np.uint8)
	img1 = Image.fromarray(img1).convert('RGB')
	img2 = Image.fromarray(img2).convert('RGB')
	img1.save('./samples/train/{}_{}_{}.png'.format(epoch,step,'real'))
	img2.save('./samples/train/{}_{}_{}.png'.format(epoch,step,'recon'))

def save_img_test(img1,img2,step,epoch):
	img1 = np.squeeze(img1)
	img2 = np.squeeze(img2)
	print(img1.shape)
	img1[img1>1.]=1.
	img1[img1<0.]=0.
	img2[img2>1.]=1.
	img2[img2<0.]=0.
	img1 = img1*255
	img2 = img2*255
	img1 = img1.astype(np.uint8)
	img2 = img2.astype(np.uint8)
	img1 = Image.fromarray(img1).convert('RGB')
	img2 = Image.fromarray(img2).convert('RGB')
	img1.save('./samples/test/{}_{}_{}.png'.format(epoch,step,'real'))
	img2.save('./samples/test/{}_{}_{}.png'.format(epoch,step,'recon'))

def save_img_res(img1,img2,img3,step,epoch):
	img1 = np.squeeze(img1)
	img2 = np.squeeze(img2)
	img3 = np.squeeze(img3)
	print(img1.shape)
	img1[img1>1.]=1.
	img1[img1<0.]=0.
	img2[img2>1.]=1.
	img2[img2<0.]=0.
	img3[img3>1.]=1.
	img3[img3<0.]=0.
	img1 = img1*255
	img2 = img2*255
	img3 = img3*255
	img1 = img1.astype(np.uint8)
	img2 = img2.astype(np.uint8)
	img3 = img3.astype(np.uint8)
	img1 = Image.fromarray(img1).convert('RGB')
	img2 = Image.fromarray(img2).convert('RGB')
	img3 = Image.fromarray(img3).convert('RGB')
	img1.save('./samples/test/{}_{}_{}.png'.format(epoch,step,'input_res'))
	img2.save('./samples/test/{}_{}_{}.png'.format(epoch,step,'output_res'))
	img3.save('./samples/test/{}_{}_{}.png'.format(epoch,step,'output_last_scale'))