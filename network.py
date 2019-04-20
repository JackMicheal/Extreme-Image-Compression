import tensorflow as tf
from tensorflow.python.ops.init_ops import Constant

class res_block(tf.keras.Model):
    def __init__(self,channels=128):
        super(res_block, self).__init__()
        
        self.conv0 = tf.keras.layers.Conv2D(channels, (3, 3),padding='same',use_bias=True)
        self.conv1 = tf.keras.layers.Conv2D(channels, (3, 3),padding='same',use_bias=True)
        self.conv2 = tf.keras.layers.Conv2D(channels, (3, 3),padding='same',use_bias=True)
    
    def call(self,x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        return x + y

class residual_block(tf.keras.Model):
    def __init__(self,channels=128,kernel_size=3,strides=1,actv=tf.nn.relu):
        super(residual_block, self).__init__()

        self.conv0 = tf.keras.layers.Conv2D(channels, (3, 3), (1, 1), padding='VALID',use_bias=True)
        self.norm0 = tf.contrib.layers.instance_norm
        self.actv0 = actv
        self.conv1 = tf.keras.layers.Conv2D(channels, (3, 3), (1, 1), padding='VALID',use_bias=True)
        self.norm1 = tf.contrib.layers.instance_norm
        self.actv1 = actv        

    def call(self, x):
        p = int((kernel_size-1)/2)
        identity_data = x

        x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        
        output = self.conv0(x)
        output = self.norm0(output)
        output = self.actv0(output)

        output = tf.pad(output, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')        
        
        output = self.conv1(output)
        output = self.norm1(output)
        output = self.actv1(output)
        # output *= 0.1
        output = tf.add(output, identity_data)
        return output

class conv_block(tf.keras.Model):
    def __init__(self,filters,kernel_size=3,strides=1,padding='SAME',actv=tf.nn.relu):
        super(conv_block, self).__init__()
        # in_kwargs = {'center':True, 'scale': True}
        self.conv0 = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding=padding)
        # self.conv0 = tf.layers.conv2d(filters,kernel_size,strides,padding,activation=None)
        # self.norm = tf.contrib.layers.instance_norm(**in_kwargs)
        self.actv = actv

    def call(self, x):
        in_kwargs = {'center':True, 'scale': True}
        output = self.conv0(x)
        output = tf.contrib.layers.instance_norm(output,**in_kwargs)
        # output = self.norm(output)
        output = self.actv(output)
        return output    

class upsample_block(tf.keras.Model):
    def __init__(self,filters,kernel_size=3,strides=2,padding='SAME',actv=tf.nn.relu):
        super(upsample_block, self).__init__()
        # in_kwargs = {'center':True, 'scale': True}
        self.conv0 = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding=padding)            
        # self.conv0 = tf.layers.conv2d_transpose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None)
        # self.norm = tf.contrib.layers.instance_norm(**in_kwargs)
        self.actv = actv

    def call(self, x):
        in_kwargs = {'center':True, 'scale': True}
        output = self.conv0(x)
        output = tf.contrib.layers.instance_norm(output,**in_kwargs)
        # output = self.norm(output)
        output = self.actv(output)
        return output   

class Deepcoder(tf.keras.Model):
    def __init__(self,M=4):
        super(Deepcoder, self).__init__()
        
        self.half = tf.constant(0.5)
        
        num_channels = [60,120,240,360,480]

        self.conv_in = conv_block(filters=num_channels[0],kernel_size=7,strides=1,padding='VALID')
        
        self.conv0 = conv_block(filters=num_channels[1],kernel_size=3,strides=2)
        self.conv1 = conv_block(filters=num_channels[2],kernel_size=3,strides=2)
        self.conv2 = conv_block(filters=num_channels[3],kernel_size=3,strides=2)
        self.conv3 = conv_block(filters=num_channels[4],kernel_size=3,strides=2)

        self.conv_enc = conv_block(filters=M,kernel_size=3,strides=1,padding='VALID')

        self.conv_dec = conv_block(filters=num_channels[4],kernel_size=3,strides=1,padding='VALID')

        self.res_module = tf.keras.Sequential()
        for i in range(9):
            self.res_module.add(residual_block(channels=num_channels[4]))

        self.deconv0 = upsample_block(filters=num_channels[3],kernel_size=3,strides=2)
        self.deconv1 = upsample_block(filters=num_channels[2],kernel_size=3,strides=2)
        self.deconv2 = upsample_block(filters=num_channels[1],kernel_size=3,strides=2)
        self.deconv3 = upsample_block(filters=num_channels[0],kernel_size=3,strides=2)

        self.conv_out = conv_block(filters=3,kernel_size=7,strides=1,padding='VALID',actv=tf.nn.tanh)
   
    def encoder(self, im, training,c=[10],scale=False,offset = None):
        x = tf.pad(im, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        
        x1 = self.conv_in(x) #3->num_channels
        
        x2 = self.conv0(x1) # downsample
        x3 = self.conv1(x2) # downsample        
        x4 = self.conv2(x3) # downsample        
        encoded = self.conv3(x4) # downsample
        encoded = tf.pad(encoded, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        encoded_raw = self.conv_enc(encoded)

        
        # TODO: squeeze
        if scale == True:
            encoded = self.conv_scale.squeeze(encoded)
        
        #add uniform noise
        if training == True:
            dy = tf.random_uniform(shape=tf.shape(encoded_raw),minval=-self.half,maxval=self.half)
            encoded = tf.math.add(encoded_raw,dy)
        else:
            if offset == None:
                encoded = tf.round(encoded_raw)
            else:    
                encoded = tf.round(encoded_raw - offset)
        #encoded_cut,c = self.cut(encoded,training = training,c=c)
        return encoded_raw,encoded
    
    def decoder(self, encoded, scale=False):
        
        # TODO: flatten
        if scale == True:
            encoded = self.conv_scale.flatten(encoded)
        #encoded = self.conv_scale.flatten(encoded)
        x4 = tf.pad(encoded, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')

        x4 = self.conv_dec(x4)
        
        x5 = self.deconv0(x4) #conv_transpose
        x6 = self.deconv1(x5) #conv_transpose
        x7 = self.deconv2(x6) #conv_transpose
        x8 = self.deconv3(x7) #conv_transpose  

        x8 = tf.pad(x8, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')

        out = self.conv_out(x8)
        
        return out
    
    # @tf.contrib.eager.defun
    def call(self, x, training=True,scale=False):
        encoded_raw,encoded = self.encoder(x,training,scale=scale)
        out = self.decoder(encoded,scale=scale)
        return encoded_raw,encoded,out

class discriminator(tf.keras.Model):
    def __init__(self,kernel_size=4,actv=tf.nn.leaky_relu):
        super(discriminator, self).__init__()
        self.conv_in =  tf.keras.layers.Conv2D(64, kernel_size=kernel_size, strides=2, padding='same')
        self.actv0 = actv
        # self.conv_in = tf.layers.conv2d(64, kernel_size=kernel_size, strides=2, padding='same', activation=actv)
        self.conv1 = conv_block(filters=128,kernel_size=kernel_size,strides=2,padding='same',actv=actv)
        self.conv2 = conv_block(filters=256,kernel_size=kernel_size,strides=2,padding='same',actv=actv)
        self.conv3 = conv_block(filters=512,kernel_size=kernel_size,strides=2,padding='same',actv=actv)
        # self.conv_out = tf.layers.conv2d(1, kernel_size=kernel_size, strides=1, padding='same')
        self.conv_out =  tf.keras.layers.Conv2D(1, kernel_size=kernel_size, strides=1, padding='same')
        self.actv1 = actv

    def call(self, x):
        out1 = self.conv_in(x)
        out1 = self.actv0(out1)
        out2 = self.conv1(out1)
        out3 = self.conv2(out2)
        out4 = self.conv3(out3)
        output = self.conv_out(out4)
        output = self.actv1(output)

        return output, [out1,out2,out3,out4] 

class multiscale_discriminator(tf.keras.Model):
    def __init__(self):
        super(multiscale_discriminator, self).__init__()
        self.disc1 = discriminator()
        self.disc2 = discriminator()
        self.disc3 = discriminator()
        # self.down_x2 = tf.layers.average_pooling2d(pool_size=3, strides=2, padding='same')
        # self.down_x4 = tf.layers.average_pooling2d(pool_size=3, strides=2, padding='same')

    def call(self, x):
        x2 = tf.layers.average_pooling2d(x,pool_size=3, strides=2, padding='same')
        x4 = tf.layers.average_pooling2d(x2,pool_size=3, strides=2, padding='same')
        disc, Dk = self.disc1(x)
        disc_downsampled_2, Dk_2 = self.disc2(x2)
        disc_downsampled_4, Dk_4 = self.disc3(x4)

        return disc, disc_downsampled_2, disc_downsampled_4, [Dk, Dk_2, Dk_4]

class up(tf.keras.Model):
    def __init__(self,filters=3,kernel_size=3,strides=2,padding='SAME',actv=tf.nn.tanh):
        super(up, self).__init__()
        in_kwargs = {'center':True, 'scale': True}
        self.conv0 = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding=padding) 
        # self.conv0 = tf.layers.conv2d_transpose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None)
        self.actv = actv

    def call(self, x):
        output = self.conv0(x)
        output = self.actv(output)
        return output   
