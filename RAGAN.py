import tensorflow as tf

from dataset import *
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Flatten, Dense, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, LeakyReLU, ZeroPadding2D, Activation, Dropout
from keras.layers import Reshape, Multiply, Dot, Concatenate, Lambda, Layer, Add, Masking, MaxPooling2D, Subtract, Cropping2D, UpSampling2D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.constraints import max_norm
from keras.optimizers import Adam, RMSprop
from keras.utils import multi_gpu_model
from PIL import Image
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import cosine
from scipy.ndimage import zoom
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


import tqdm
from EMA import *
from custom_losses import *
from custom_layers import *
from custom_blocks import *
from optimizer import *
#v90 final
class RAGAN(object):

    def __init__(self, model_save_dir="models"):
        self.model_save_dir = model_save_dir
        self.resl_num = 6 # (16, 16, 32, 64, 128, 256)
        self.style_size = 64 # The size of Style Code
        self.image_size = 4 * (2 ** self.resl_num) # 4 * (2 ^ 6) = 256
        self.batch_size = 8 # Set to 8 if able.
        self.domain_size = 2 # Male and Female
        self.latent_size = 16 # The input size of Mapping Network

    def define_models(self):
        self.build_components()

        self.build_G_model()
        self.build_M_model()
        self.build_D_model()
        self.build_S_model()

        self.G_model.summary()
        self.M_model.summary()
        self.D_model.summary()
        self.S_model.summary()

    # Training Graph
    def define_training_models(self):
        self.div_alpha = K.variable(1.0)
        self.IDweight = K.variable(0.0)
        self.same_weight = K.variable(0)
        self.test_code = [np.tile(np.random.normal(0.0, 1.0, size=(self.latent_size)), (4, 1)) for _ in range(3)]
        self.test_src_data = get_val_data(image_size=self.image_size)
        #self.test_ref_data = get_ref_data(set_type="val", image_size=self.image_size)

        multipliers = {"MNFC": 0.01}
        
        D_optimizer = AdamW(learning_rate=0.0001, beta_1=0, beta_2=0.99, epsilon=1e-8, weight_decay=1e-4)
        S_optimizer = AdamW(learning_rate=0.0001, beta_1=0, beta_2=0.99, epsilon=1e-8, weight_decay=1e-4)
        G_optimizer = AdamW(learning_rate=0.0001, beta_1=0, beta_2=0.99, epsilon=1e-8, weight_decay=1e-4, multipliers=multipliers, debug_verbose=True)

        self.build_train_D(optimizer=D_optimizer)
        self.build_train_G(optimizer=G_optimizer)
        self.build_train_S(optimizer=S_optimizer)

    def load_model(self, G_path=None, M_path=None, D_path=None, S_path=None):

        if G_path:
            self.G_model.load_weights(os.path.join(self.model_save_dir, G_path))
        else:
            print("No weights for Generator")

        if M_path:
            self.M_model.load_weights(os.path.join(self.model_save_dir, M_path))
        else:
            print("No weights for Mapping Network")

        if D_path:
            self.D_model.load_weights(os.path.join(self.model_save_dir, D_path))
        else:
            print("No weights for Disciminator")

        if S_path:
            self.S_model.load_weights(os.path.join(self.model_save_dir, S_path))
        else:
            print("No weights for Style Encoder")

    def save_model(self, G_path, M_path, D_path, S_path):
        self.G_model.save_weights(os.path.join(self.model_save_dir, G_path))
        self.M_model.save_weights(os.path.join(self.model_save_dir, M_path))       
        self.D_model.save_weights(os.path.join(self.model_save_dir, D_path)) 
        self.S_model.save_weights(os.path.join(self.model_save_dir, S_path)) 

    def build_components(self):

        G_channel_in = 32
        D_channel_in = 32
        D_channels = [64, 128, 256, 512, 512, 512]
        D_channel_out = 512
        S_channel_out = 512

        self.input_image = Input(shape=(self.image_size, self.image_size, 3), name="Input-Image")
        self.input_class = Input(shape=(self.domain_size, ), name="Input-Class")
        self.input_mask = Input(shape=(self.image_size, self.image_size, 2), name="Input-Mask")

        self.G_fRGB = Conv2D(filters=G_channel_in, kernel_size=1, padding="same", name="G-From-Image")
        self.D_fRGB = Conv2D(filters=D_channel_in, kernel_size=1, padding="same", name="D-From-Image")
        self.S_fRGB = Conv2D(filters=G_channel_in, kernel_size=1, padding="same", name="S-From-Image")

        self.G_tRGB = Sequential(name="To-Image")
        self.G_tRGB.add(InstanceNormalization())
        self.G_tRGB.add(LeakyReLU(0.2))
        self.G_tRGB.add(Conv2D(filters=3, kernel_size=1))
        """
        self.G_to_imask = Sequential(name="To-Imask")
        self.G_to_imask.add(BatchNormalization())
        self.G_to_imask.add(ReLU())
        self.G_to_imask.add(Conv2D(filters=1, kernel_size=1, activation="sigmoid"))
        
        self.G_to_cmask = Sequential(name="To-Cmask")
        self.G_to_cmask.add(BatchNormalization())
        self.G_to_cmask.add(ReLU())
        self.G_to_cmask.add(Conv2D(filters=1, kernel_size=1, activation="sigmoid"))
        """
        self.D_blocks = []
        self.S_blocks = []

        self.DN_blocks = [
            IN_block(x_shape=(256, 256,  32), inp_channel= 32, out_channel= 64, name="G-Block-IN-DN-1", res_add=True, dn_sample=True),
            IN_block(x_shape=(128, 128,  64), inp_channel= 64, out_channel=128, name="G-Block-IN-DN-2", res_add=True, dn_sample=True),
            IN_block(x_shape=( 64,  64, 128), inp_channel=128, out_channel=256, name="G-Block-IN-DN-3", res_add=True, dn_sample=True),
            IN_block(x_shape=( 32,  32, 256), inp_channel=256, out_channel=512, name="G-Block-IN-DN-4", res_add=True, dn_sample=True)
        ]
        self.DS_blocks = [
            IN_block(x_shape=(256, 256,  32), inp_channel= 32, out_channel= 64, name="G-Block-IN-DN-1", res_add=True, dn_sample=True),
            IN_block(x_shape=(128, 128,  64), inp_channel= 64, out_channel=128, name="G-Block-IN-DN-2", res_add=True, dn_sample=True),
            IN_block(x_shape=( 64,  64, 128), inp_channel=128, out_channel=256, name="G-Block-IN-DN-3", res_add=True, dn_sample=True),
            IN_block(x_shape=( 32,  32, 256), inp_channel=256, out_channel=512, name="G-Block-IN-DN-4", res_add=True, dn_sample=True)
        ]

        self.IN_blocks = [
            IN_block(x_shape=( 16,  16, 512), inp_channel=512, out_channel=512, name="G-Block-IN-1", res_add=True),
            IN_block(x_shape=( 16,  16, 512), inp_channel=512, out_channel=512, name="G-Block-IN-2", res_add=True),
            AN_block(x_shape=( 16,  16, 512), s_shape=(self.style_size, ), inp_channel=512, out_channel=512, name="G-Block-AN-1"),
            AN_block(x_shape=( 16,  16, 512), s_shape=(self.style_size, ), inp_channel=512, out_channel=512, name="G-Block-AN-2"),
        ]

        self.UP_blocks = [
            AN_block(x_shape=( 16,  16, 512), s_shape=(self.style_size, ), c_shape=( 32,  32, 256), inp_channel=512, out_channel=256, name="G-Block-AN-UP-1", up_sample=True),
            AN_block(x_shape=( 32,  32, 256), s_shape=(self.style_size, ), c_shape=( 64,  64, 128), inp_channel=256, out_channel=128, name="G-Block-AN-UP-2", up_sample=True),
            AN_block(x_shape=( 64,  64, 128), s_shape=(self.style_size, ), c_shape=(128, 128,  64), inp_channel=128, out_channel= 64, name="G-Block-AN-UP-3", up_sample=True),
            AN_block(x_shape=(128, 128,  64), s_shape=(self.style_size, ), c_shape=(256, 256,  32), inp_channel= 64, out_channel= 32, name="G-Block-AN-UP-4", up_sample=True)
        ]
        """
        self.SG_blocks = [
            SN_block(x_shape=( 16,  16, 512), m_shape=( 32,  32, 256), inp_channel=512, out_channel=256, name="G-Seg-Block-1"),
            SN_block(x_shape=( 32,  32, 256), m_shape=( 64,  64, 128), inp_channel=256, out_channel=128, name="G-Seg-Block-2"),
            SN_block(x_shape=( 64,  64, 128), m_shape=(128, 128,  64), inp_channel=128, out_channel= 64, name="G-Seg-Block-3"),
            SN_block(x_shape=(128, 128,  64), m_shape=(256, 256,  32), inp_channel= 64, out_channel= 32, name="G-Seg-Block-4")
        ]
        """
        for idx in range(self.resl_num):

            x_dim = self.image_size // (2 ** idx)
            inp_channel = D_channel_in if idx == 0 else D_channels[idx - 1]
            out_channel = D_channels[idx]

            D_block = DC_block(x_shape=(x_dim, x_dim, inp_channel), inp_channel=inp_channel, out_channel=out_channel, name=f"D-Block-{idx + 1}")

            if idx < 4:
                S_block = self.DS_blocks[idx]
            else:
                S_block = DC_block(x_shape=(x_dim, x_dim, inp_channel), inp_channel=inp_channel, out_channel=out_channel, name=f"S-Block-{idx + 1}")

            self.S_blocks.append(S_block)
            self.D_blocks.append(D_block)

        self.D_final_block = Sequential(name="D-Final_block")
        self.D_final_block.add(LeakyReLU(0.2))
        self.D_final_block.add(Conv2D(filters=D_channel_out, kernel_size=4, padding="valid"))
        self.D_final_block.add(LeakyReLU(0.2))
        self.D_final_block.add(Conv2D(filters=self.domain_size, kernel_size=1, padding="valid"))
        self.D_final_block.add(Reshape((self.domain_size, )))

        self.S_final_block = Sequential(name="S-Block-Final")
        self.S_final_block.add(LeakyReLU(0.2))
        self.S_final_block.add(Conv2D(filters=S_channel_out, kernel_size=4, padding="valid"))
        self.S_final_block.add(LeakyReLU(0.2))

    def build_G_model(self): 

        input_styles = [
            Input(shape=(self.style_size, ), name=f"Input-Style-0"),
            Input(shape=(self.style_size, ), name=f"Input-Style-1"),
            Input(shape=(self.style_size, ), name=f"Input-Style-2"),
            Input(shape=(self.style_size, ), name=f"Input-Style-3"),
            Input(shape=(self.style_size, ), name=f"Input-Style-4"),
            Input(shape=(self.style_size, ), name=f"Input-Style-5"),
        ]

        x = self.G_fRGB(self.input_image)

        catches = []

        for i, block in enumerate(self.DN_blocks):
            catches.append(x)
            x = block(x)
            
        catches.reverse()

        #s = x

        #for i, block in enumerate(self.SG_blocks):
        #    s = block([s, catches[i]])

        #out_imask = self.G_to_imask(s)
        #out_cmask = self.G_to_cmask(s)

        
        def expand_dimension_key(in_mask):
          out_mask = in_mask[:,:,:,0]
          out_mask = K.expand_dims(out_mask,axis = -1)
          return out_mask
        def expand_dimension_coarse(in_mask):
          out_mask = in_mask[:,:,:,1]
          out_mask = K.expand_dims(out_mask,axis = -1)
          return out_mask
        imask = Lambda(expand_dimension_key)(self.input_mask)
        cmask = Lambda(expand_dimension_coarse)(self.input_mask)
        
        imasks = []

        for i in range(len(catches)):
            imasks.append(imask)
            imask = AveragePooling2D(name=f"AvgPool-I-Mask-{i + 1}")(imask)
            cmask = AveragePooling2D(name=f"AvgPool-C-Mask-{i + 1}")(cmask)

        imasks.reverse()

        x = Multiply(name="Mul-C-Mask")([x, cmask])

        for i, block in enumerate(self.IN_blocks):

            if i < 2:
                x = block(x)
            else:
                w = input_styles[i - 2]
                x = block([x, w])

        for i, block in enumerate(self.UP_blocks):
            w = input_styles[i + 2]

            catch = catches[i]
            imask = imasks[i]

            x = block([x, w, Multiply(name=f"Mul-I-Mask-{i + 1}")([catch, imask])])

        out_image = self.G_tRGB(x)
        #out_masks = Concatenate(name="Out-Masks", axis=-1)([out_imask, out_cmask])

        self.G_model = Model(inputs=[self.input_image,self.input_mask] + input_styles, outputs=[out_image], name="G_Model")

    def build_S_model(self):

        input_class = self.input_class

        x = self.S_fRGB(self.input_image)

        for idx, block in enumerate(self.S_blocks):
            x = block(x)

        x = self.S_final_block(x)

        ReshapeLayer = Reshape((self.domain_size, self.style_size), name="Reshape")
        style_outs = []

        for jdx in range(self.resl_num):

            branch_outs = []

            for idx in range(self.domain_size):
                branch = Sequential(name=f"Branch-{idx}-{jdx}")
                branch.add(Flatten())
                branch.add(Dense(self.style_size, kernel_initializer="he_uniform"))
                branch_outs.append(branch(x))

            out_style = Concatenate(name=f"Stack-Branch-{jdx}")(branch_outs)
            out_style = ReshapeLayer(out_style)
            out_style = Dot(axes=1, name=f"Select-Branch-{jdx}")([input_class, out_style])
            style_outs.append(out_style)

        self.S_model = Model([input_class, self.input_image], style_outs, name="S_Model")

    def build_M_model(self):
        # The name must contains "MNFC to Apply multiplier learning rate. #line 54"
        input_z = Input(name="Latent-Input", shape=[self.latent_size])
        input_class = Input(shape=(self.domain_size, ), name="Input-Class")

        shared_block = Sequential(name="S-MNFC")
        shared_block.add(Dense(512, kernel_initializer="he_uniform", name="FC-1"))
        shared_block.add(ReLU())
        shared_block.add(Dense(512, kernel_initializer="he_uniform", name="FC-2"))
        shared_block.add(ReLU())
        shared_block.add(Dense(512, kernel_initializer="he_uniform", name="FC-3"))
        shared_block.add(ReLU())
        shared_block.add(Dense(512, kernel_initializer="he_uniform", name="FC-4"))
        shared_block.add(ReLU())

        x = shared_block(input_z)

        unshared_branches = []

        for idx in range(self.domain_size):
            branch = Sequential(name=f"U-MNFC-{idx}")
            branch.add(Dense(512, kernel_initializer="he_uniform", name="FC-1"))
            branch.add(ReLU())
            branch.add(Dense(512, kernel_initializer="he_uniform", name="FC-2"))
            branch.add(ReLU())
            branch.add(Dense(512, kernel_initializer="he_uniform", name="FC-3"))
            branch.add(ReLU())
            unshared_branches.append(branch)

        ReshapeLayer = Reshape((self.domain_size, self.style_size), name="Reshape")

        style_outs = []

        for jdx in range(self.resl_num):

            branch_outs = []

            for idx in range(self.domain_size):
                branch = unshared_branches[idx]
                branch_outs.append(Dense(self.style_size, kernel_initializer="he_uniform", name=f"MNFC-{idx}-{jdx}")(branch(x)))

            out_style = Concatenate(name=f"Stack-Branch-{jdx}")(branch_outs)
            out_style = ReshapeLayer(out_style)
            out_style = Dot(axes=1, name=f"Select-Branch-{jdx}")([input_class, out_style])
            style_outs.append(out_style)

        self.M_model = Model([input_z, input_class], style_outs, name="M_Model")

    def build_D_model(self):

        input_class = self.input_class

        x = self.D_fRGB(self.input_image)

        for block in self.D_blocks:
            x = block(x)

        out_real = self.D_final_block(x)
        out_real = Dot(axes=1, name="Branch-Out")([input_class, out_real])

        self.D_model = Model([input_class, self.input_image], [out_real], name="D_Model")

    # The Training Graph of Discriminator
    def build_train_D(self, optimizer):

        self.D_model.trainable = True

        for layer in self.D_model.layers:
            layer.trainable = True

        real_image = Input(shape=(self.image_size, self.image_size, 3))
        fake_image = Input(shape=(self.image_size, self.image_size, 3))

        real_class = Input(shape=(self.domain_size, ))
        fake_class = Input(shape=(self.domain_size, ))

        with tf.device("/gpu:0"):
          out_src_real = self.D_model([real_class, real_image])

        with tf.device("/gpu:1"):
          out_src_fake = self.D_model([fake_class, fake_image])
            
        gp_loss = gradient_l1_penalty_loss(real_image, out_src_real)

        self.train_D = Model([real_class, fake_class, real_image, fake_image], [out_src_real, out_src_fake])
        self.train_D.add_loss( 1 * gp_loss)
        self.train_D.compile(loss=[BCE_Loss, BCE_Loss], optimizer=optimizer, loss_weights=[1, 1])
        self.train_D.add_metric(gp_loss, name="gp")

    # The Training Graph of Latent Synthesis
    def build_train_G(self, optimizer):

        self.D_model.trainable = False

        for layer in self.D_model.layers:
            layer.trainable = False

        inp_image = Input(shape=(self.image_size, self.image_size, 3))
        inp_mask = Input(shape=(self.image_size, self.image_size, 2))
        inp_class = Input(shape=(self.domain_size, ))
        fak_class = Input(shape=(self.domain_size, ))
        fak_style = Input(shape=(self.latent_size, ))
        #inp_masks = Input(shape=(self.image_size, self.image_size, 2))

        with tf.device("/gpu:0"):
          wf = self.M_model([fak_style, fak_class])
          out_image = self.G_model([inp_image, inp_mask] + wf)
          wo = self.S_model([fak_class, out_image])
          out_src_fake = self.D_model([fak_class, out_image])

        with tf.device("/gpu:1"):
          wi = self.S_model([inp_class, inp_image])
          cyc_image = self.G_model([out_image, inp_mask] + wi)

        style_loss = K.mean(K.abs(K.concatenate(wf , axis=-1) - K.concatenate(wo , axis=-1)))
        
        face_image = out_image 
        crop_face = Cropping2D(cropping=((94, 50), (69, 75)))(face_image) # 112
        up_face = UpSampling2D(size=(2, 2),interpolation="bilinear")(crop_face)
        def subtract_fix(up_face):
            r = up_face[:,:,:,0]-0.3588
            g = up_face[:,:,:,1]-0.4074
            b = up_face[:,:,:,2]-0.5141
            r = K.expand_dims(r,axis = -1)
            g = K.expand_dims(g,axis = -1)
            b = K.expand_dims(b,axis = -1)
            out_face = K.concatenate([r,g,b],axis = -1)
            return out_face
        out_face = Lambda(subtract_fix)(up_face)
        
        
        
        out_masks = inp_mask
        def expand_repeat(out_masks):
          out_mask = out_masks[:,:,:,1]
          out_mask_e = K.expand_dims(out_mask,axis = -1)
          #out_mask_r = K.repeat_elements(out_mask_e,3,axis = -1) #會出現graph sorted error
          out_mask_r = K.tile(out_mask_e,[1,1,1,3])
          return out_mask_r
        out_mask_r = Lambda(expand_repeat)(out_masks)
        cyc_out_face = Multiply(name="Recog_out")([cyc_image, out_mask_r])
        
        #print(out_face.shape)
        self.train_G = Model([inp_image, inp_class, inp_mask, fak_class, fak_style], [out_src_fake, cyc_out_face, out_image,out_face])  #y = y_pred
        self.train_G.add_loss(3 * style_loss)
        #self.train_G.add_loss(0.01 * ID_loss_value)
        
        self.train_G.compile(loss=[BCE_Loss, reconstruction_loss, reconstruction_loss, ID_loss], loss_weights=[1, 5, -self.div_alpha, self.IDweight], optimizer=optimizer)
        self.train_G.add_metric(style_loss, name="style_loss")
        #self.train_G.add_metric(ID_loss_value, name="ID_loss")

    # The Training Graph of Reference Synthesis
    def build_train_S(self, optimizer):

        self.D_model.trainable = False

        for layer in self.D_model.layers:
            layer.trainable = False

        self.S_model.trainable = False

        for layer in self.S_model.layers:
            layer.trainable = False

        inp_image = Input(shape=(self.image_size, self.image_size, 3))
        ref_image = Input(shape=(self.image_size, self.image_size, 3))
        inp_mask = Input(shape=(self.image_size, self.image_size, 2))
        inp_class = Input(shape=(self.domain_size, ))
        ref_class = Input(shape=(self.domain_size, ))
        same_recon = Input(shape=(1,))
        #inp_masks = Input(shape=(self.image_size, self.image_size, 2))

        with tf.device("/gpu:1"):
          wr = self.S_model([ref_class, ref_image])
          out_image = self.G_model([inp_image, inp_mask] + wr)
          wo = self.S_model([ref_class, out_image])
          out_src_fake = self.D_model([ref_class, out_image])

        with tf.device("/gpu:0"):
          wi = self.S_model([inp_class, inp_image])
          cyc_image = self.G_model([out_image, inp_mask] + wi)

        face_image = out_image
        same_out_image = out_image
        style_loss = K.mean(K.abs(K.concatenate(wr , axis=-1) - K.concatenate(wo , axis=-1)))
        
        # triplet loss
        ap = K.abs(K.concatenate(wr , axis=-1) - K.concatenate(wo , axis=-1))
        an = K.abs(K.concatenate(wi , axis=-1) - K.concatenate(wo , axis=-1))
        
        m = 0.5
        zero = K.zeros_like(ap)
        ones = K.ones_like(ap)
        margin = ones * m
        t = K.maximum(zero, margin+(ap-an))
        triplet_loss = K.mean(t)
        
        #same_recon_loss = K.mean(same_recon,axis=0)*K.mean(K.square(K.abs(same_out_image-inp_image)),axis=-1)
        
        crop_face = Cropping2D(cropping=((94, 50), (69, 75)))(face_image) # 112
        up_face = UpSampling2D(size=(2, 2),interpolation="bilinear")(crop_face)
        
        def subtract_fix(up_face):
            r = up_face[:,:,:,0]-0.3588
            g = up_face[:,:,:,1]-0.4074
            b = up_face[:,:,:,2]-0.5141
            r = K.expand_dims(r,axis = -1)
            g = K.expand_dims(g,axis = -1)
            b = K.expand_dims(b,axis = -1)
            out_face = K.concatenate([r,g,b],axis = -1)
            return out_face
        out_face = Lambda(subtract_fix)(up_face)
        
        out_masks = inp_mask
        def expand_repeat(out_masks):
          out_mask = out_masks[:,:,:,1]
          out_mask_e = K.expand_dims(out_mask,axis = -1)
          #out_mask_r = K.repeat_elements(out_mask_e,3,axis = -1)  # K.tile
          out_mask_r = K.tile(out_mask_e,[1,1,1,3])
          return out_mask_r
        out_mask_r = Lambda(expand_repeat)(out_masks)
        cyc_out_face = Multiply(name="Recog_out")([cyc_image, out_mask_r])
        
        self.train_S = Model([inp_image, inp_class, inp_mask, ref_image, ref_class,same_recon], [out_src_fake, cyc_out_face, out_image, out_face])
        self.train_S.add_loss(3 * style_loss)
        self.train_S.add_loss(10 * triplet_loss)
        #self.train_S.add_loss(self.same_weight * same_recon_loss)
        self.train_S.compile(loss=[BCE_Loss, reconstruction_loss, reconstruction_loss, ID_loss], loss_weights=[1, 5, -self.div_alpha, self.IDweight], optimizer=optimizer)
        self.train_S.add_metric(style_loss, name="style_loss")
        self.train_S.add_metric(triplet_loss, name="triplet_loss")
        #self.train_S.add_metric(same_recon_loss, name="same_recon_loss")
        
    def train(self, epoches, start_epoch=0):
        
        real =  np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        
        # Add tensorboard
        #logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        #log_writer = tf.summary.create_file_writer(logdir)

        EMA_model = ExponentialMovingAverage([self.G_model, self.M_model, self.S_model])

        if start_epoch != 0:
            G_path = f"StarGANv2_G_weights-{start_epoch - 1}.hdf5"
            M_path = f"StarGANv2_M_weights-{start_epoch - 1}.hdf5"
            D_path = f"StarGANv2_D_weights-{start_epoch - 1}.hdf5"
            S_path = f"StarGANv2_S_weights-{start_epoch - 1}.hdf5"
            self.load_model(G_path, M_path, D_path, S_path)

            EG_path = f"EMA_G-{start_epoch - 1}.hdf5"
            EM_path = f"EMA_M-{start_epoch - 1}.hdf5"
            ES_path = f"EMA_S-{start_epoch - 1}.hdf5"
            EMA_model.EMA_load(self.model_save_dir, [EG_path, EM_path, ES_path])

        for epoch in range(start_epoch, epoches):
            self.train_src_data = get_src_data(set_type="train", image_size=self.image_size)
            self.train_ref_data = get_ref_data(set_type="train", image_size=self.image_size)
            
            src_data_iter = get_src_loader(self.train_src_data, self.batch_size, shuffle=True, flip=True)
            ref_data_iter = get_ref_loader(self.train_ref_data, self.batch_size, shuffle=True, flip=True)

            train_log = tqdm.tqdm(total=28000, bar_format="{desc}{percentage:3.0f}%|{bar:40} {r_bar}", desc=f"Epoch[{epoch + 1:03d}]", position=0)
            d_log = tqdm.tqdm(total=0, bar_format="{desc}", position=1)
            g_log = tqdm.tqdm(total=0, bar_format="{desc}", position=2)
            s_log = tqdm.tqdm(total=0, bar_format="{desc}", position=3)
            if (epoch == 0):
               K.set_value(self.IDweight, 1.5)
            elif (epoch == 5):               
               K.set_value(self.IDweight, 1.0)
            elif (epoch == 10):               
               K.set_value(self.IDweight, 0.5)
            elif (epoch == 12):          
               K.set_value(self.IDweight, 0.0)
            elif (epoch == 14):          
               K.set_value(self.IDweight, 0.0)
            elif (epoch == 18):
               K.set_value(self.IDweight, 0.0)
            elif (epoch > 18):
               K.set_value(self.IDweight, 0.0)
            
            """
            elif (epoch == 14):
               K.set_value(self.IDweight, 0.1)
               K.set_value(self.same_weight, 0.5)
            elif (epoch == 18):
               K.set_value(self.IDweight, 0.0)
               K.set_value(self.same_weight, 0.9)
            """   
            D_loss_list = []
            G_loss_list = []
            S_loss_list = []
            for iter, (src_data, ref_data) in enumerate(zip(src_data_iter, ref_data_iter)):
                
                same_recon = np.zeros((self.batch_size,1))
                real_images, real_domains, real_imasks, progress = src_data
                # modified
                if np.random.rand() > 2:
                   ref1_images, ref2_images, reff_domains = ref_data
                   ref1_images = real_images
                   ref2_domains = reff_domains 
                   ref1_domains = real_domains
                   same_recon = np.ones((self.batch_size,1))
                else :
                   ref1_images, ref2_images, ref1_domains = ref_data
                   ref2_domains = ref1_domains

                #print("This is real imasks")
                #print(real_imasks[0,100:160,100:160,1])
                #  multiply the coarse mask
                real_imask = real_imasks[:,:,:,1]
                real_imask_e = np.expand_dims(real_imask,axis = -1)
                real_imask_r = np.repeat(real_imask_e,3,axis = -1)
                cyc_real_images = np.multiply(real_images,real_imask_r)
                
                croped_face = real_images[:,94:206,69:181,:]  # 112*112
                zoom_face = np.zeros((self.batch_size,224,224,3))
                inp_face = np.zeros((self.batch_size,224,224,3))
                for b in range(0, self.batch_size):
                  zoom_face[b,:,:,0] = zoom(croped_face[b,:,:,0], 2, order=1) # bilinear
                  zoom_face[b,:,:,1] = zoom(croped_face[b,:,:,1], 2, order=1) # bilinear
                  zoom_face[b,:,:,2] = zoom(croped_face[b,:,:,2], 2, order=1) # bilinear
                  
                inp_face[:,:,:,0] = zoom_face[:,:,:,0]-0.3588
                inp_face[:,:,:,1] = zoom_face[:,:,:,1]-0.4074
                inp_face[:,:,:,2] = zoom_face[:,:,:,2]-0.5141
                
                
                #inp_face = real_images[:,0:224,0:224,:]   (94, 228), (60, 194)
                
                #print("train inp")
                #print(inp_face.shape)
                
                alpha = (epoch + progress) / epoches
                fake_domains = random_domains(self.batch_size, self.domain_size)

                K.set_value(self.div_alpha, 1 - alpha) # Lambda of diversity loss, 1 linear decay to 0.

                z1 = np.random.normal(0.0, 1.0, size=(self.batch_size, self.latent_size))
                z2 = np.random.normal(0.0, 1.0, size=(self.batch_size, self.latent_size))

                fake_images = self.G_model.predict([real_images, real_imasks] + self.M_model.predict([z1, fake_domains]))
                #print(fake_imask.shape)  shape = (8,256,256,2)  (8,256,256,0)是key mask (8,256,256,1)是coarse mask 
                #print(fake_imask[0])
                #print("This is fake imasks")
                #print(fake_imask[0,:,:,1])
                styl_images = self.G_model.predict([real_images, real_imasks] + self.S_model.predict([ref1_domains, ref1_images]))
                #print("This is style imasks")
                #print(styl_imask[0,:,:,1])

                D_loss_A = self.train_D.train_on_batch(x=[real_domains, fake_domains, real_images, fake_images], y=[real, fake])
                D_loss_B = self.train_D.train_on_batch(x=[real_domains, ref1_domains, real_images, styl_images], y=[real, fake])
                D_loss = (np.array(D_loss_A) + np.array(D_loss_B)) * 0.5

                fdiv_images = self.G_model.predict([real_images, real_imasks] + self.M_model.predict([z2, fake_domains]))
                G_loss = self.train_G.train_on_batch(x=[real_images, real_domains, real_imasks, fake_domains, z1], y=[real, cyc_real_images, fdiv_images, inp_face]) #y = y_true

                sdiv_images = self.G_model.predict([real_images, real_imasks] + self.S_model.predict([ref2_domains, ref2_images]))
                S_loss = self.train_S.train_on_batch(x=[real_images, real_domains, real_imasks, ref1_images, ref1_domains,same_recon], y=[real, cyc_real_images, sdiv_images, inp_face])

                EMA_model.update()

                train_log.update(n=self.batch_size)
                #if iter < 5:
                d_log.set_description_str(f"D/loss = [{D_loss[0]: 8.3f}], real = [{D_loss[1]: 8.3f}], loss_fake = [{D_loss[2]: 8.3f}], loss_gp   = [{D_loss[3]: 8.3f}], alpha = [{K.get_value(self.div_alpha): .4f}]")
                g_log.set_description_str(f"G/loss = [{G_loss[0]: 8.3f}], real = [{G_loss[1]: 8.3f}], cycl = [{G_loss[2]: 8.3f}], divr = [{G_loss[3]: 8.3f}], ID = [{G_loss[4]: 8.3f}], styl = [{G_loss[5]: 8.5f}]")
                s_log.set_description_str(f"S/loss = [{S_loss[0]: 8.3f}], real = [{S_loss[1]: 8.3f}], cycl = [{S_loss[2]: 8.3f}], divr = [{S_loss[3]: 8.3f}], ID = [{S_loss[4]: 8.3f}], styl = [{S_loss[5]: 8.5f}], trip = [{S_loss[6]: 8.5f}]")

                D_loss_list.append([D_loss[0],D_loss[1],D_loss[2],D_loss[3]])
                G_loss_list.append([G_loss[0],G_loss[1],G_loss[2],G_loss[3],G_loss[4],G_loss[5]])
                S_loss_list.append([S_loss[0],S_loss[1],S_loss[2],S_loss[3],S_loss[4],S_loss[5],S_loss[6]])
               # with log_writer.as_default():
               #    tf.summary.scalar("D loss",float(D_loss[0]),step=iter)
               #    tf.summary.scalar("G loss",float(G_loss[0]),step=iter)                   
               #    tf.summary.scalar("S loss",float(S_loss[0]),step=iter)


                if iter % 350 == 0:
                    save_name = f"{epoch}_{iter // 350}"
                    self.test(folder="results/SimpleTest", simple=3, simple_name=save_name) # Test 1 Image using original models
                    EMA_model.EMA_test(lambda: self.test(folder="results/EMA_Test", simple=3, simple_name=save_name)) # Test 1 Image using EMA models

            train_log.close()
            d_log.close()
            g_log.close()
            s_log.close()

            G_path = f"StarGANv2_G_weights-{epoch}.hdf5"
            M_path = f"StarGANv2_M_weights-{epoch}.hdf5"
            D_path = f"StarGANv2_D_weights-{epoch}.hdf5"
            S_path = f"StarGANv2_S_weights-{epoch}.hdf5"
            
            fname_D = 'D_loss_' + str(epoch) + '.txt'
            fname_G = 'G_loss_' + str(epoch) + '.txt'
            fname_S = 'S_loss_' + str(epoch) + '.txt'
            np.savetxt(fname_D, np.row_stack(D_loss_list), fmt="%.3f")
            np.savetxt(fname_G, np.row_stack(G_loss_list), fmt="%.3f")
            np.savetxt(fname_S, np.row_stack(S_loss_list), fmt="%.3f")
    
            self.save_model(G_path, M_path, D_path, S_path) # Save original weights
            EMA_model.EMA_save(directory=self.model_save_dir, paths=[f"EMA_G-{epoch}.hdf5", f"EMA_M-{epoch}.hdf5", f"EMA_S-{epoch}.hdf5"]) # Save EMA weights

            if (epoch + 1) % 5 == 0:
                EMA_model.EMA_test(lambda: self.test(folder=f"results/{epoch + 1}"))

            #if (epoch + 1) % 15 == 0:
                #EMA_model.EMA_test(lambda: self.test_mix(folder=f"results/stylemix-{epoch + 1}")) 
                    
    def test(self, folder, batch_size=4, simple=None, simple_name=None):
        
        data_iter = get_val_loader(self.test_src_data, batch_size)

        if not os.path.exists(folder):
            os.makedirs(folder)

        for iter, val_data in enumerate(data_iter):

            if simple != None and iter != simple:
                continue

            real_images, real_labels, real_masks, reff_data = val_data
            non_array = np.ones_like(real_images[0])
            src_array = np.hstack(real_images)
            src_array = np.hstack([non_array] + [src_array])

            out_arrays = [src_array]

            for i, (reff_image, reff_label) in enumerate(reff_data):
                dup_images = np.array([reff_image] * batch_size)
                dup_labels = np.array([reff_label] * batch_size)

                out_images = self.G_model.predict([real_images, real_masks] + self.S_model.predict([dup_labels, dup_images]))

                ref_array = reff_image
                out_array = np.hstack(out_images)
                
                if simple != None:
                    out_imask = np.expand_dims(real_masks[:,:,:,i], axis=-1)
                    out_imask = np.repeat(out_imask * 2 - 1, 3, axis=-1) 
                    msk_array = np.hstack(out_imask)
                    out_arrays = out_arrays + [np.hstack([ref_array] + [out_array])] + [np.hstack([non_array] + [msk_array])]
                else:
                    out_arrays = out_arrays + [np.hstack([ref_array] + [out_array])]

            image_array = np.vstack(out_arrays)

            if simple_name:
                Image.fromarray(toImage255(image_array)).save(os.path.join(folder, f"I_{simple_name}.png"))
            else:
                Image.fromarray(toImage255(image_array)).save(os.path.join(folder, f"I{iter}.png"))

            src_arrays = [np.hstack(real_images)]
            out_arrays = []

            for i in range(3):
                z = self.test_code[i]
                y1 = np.tile(np.array([1, 0]), (batch_size, 1))
                y2 = np.tile(np.array([0, 1]), (batch_size, 1))

                out_imgs1 = self.G_model.predict([real_images, real_masks] + self.M_model.predict([z, y1]))
                out_imgs2 = self.G_model.predict([real_images, real_masks] + self.M_model.predict([z, y2]))
                out_array1 = [np.hstack(out_imgs1)]
                out_array2 = [np.hstack(out_imgs2)]
                out_arrays = out_array1 + out_arrays + out_array2

            image_array = np.vstack(src_arrays + out_arrays)

            if simple_name:
                Image.fromarray(toImage255(image_array)).save(os.path.join(folder, f"M_{simple_name}.png"))
            else:
                Image.fromarray(toImage255(image_array)).save(os.path.join(folder, f"M{iter}.png"))
    """
    def test_mix(self, folder, batch_size=4):

        src_data_iter = get_val_loader(list(reversed(self.test_src_data)), batch_size)
        ref_data_iter = get_ref_loader(self.test_ref_data, batch_size)

        if not os.path.exists(folder):
            os.makedirs(folder)

        for iter, (src_data, ref_data)  in enumerate(zip(src_data_iter, ref_data_iter)):

            real_C, domn_C, _ = src_data
            reff_A, reff_B, label = ref_data

            reff_A = np.array([reff_A[0]] * 4)
            reff_B = np.array([reff_B[0]] * 4)

            code_A = self.S_model.predict([label, reff_A])
            code_B = self.S_model.predict([label, reff_B])

            out_arrays = [np.vstack(real_C)] + [np.vstack(reff_A)] + [np.vstack(reff_B)]

            for idx in range(len(code_A)):
                code_M = code_B[:idx] + code_A[idx:]
                out_images, out_imasks = self.G_model.predict([real_C] + code_M)
                out_arrays = out_arrays + [np.vstack(out_images)]

            out_arrays = np.hstack(out_arrays)
            save_path = os.path.join(folder, f"{iter}-F-A.png")
            Image.fromarray(toImage255(out_arrays)).save(save_path)

            out_arrays = [np.vstack(real_C)] + [np.vstack(reff_B)] + [np.vstack(reff_A)]

            for idx in range(len(code_B)):
                code_M = code_A[:idx] + code_B[idx:]
                out_images, out_imasks = self.G_model.predict([real_C] + code_M)
                out_arrays = out_arrays + [np.vstack(out_images)]

            out_arrays = np.hstack(out_arrays)
            save_path = os.path.join(folder, f"{iter}-F-B.png")
            Image.fromarray(toImage255(out_arrays)).save(save_path)
    """
def random_domains(batch_size, domain_size):
    domains = np.zeros(shape=(batch_size, domain_size))

    for domain in domains:
        domain[np.random.choice([0, 1])] = 1

    return domains

def toImage255(image):
    image = np.clip(image * 0.5 + 0.5, 0, 1)
    return (image * 255).astype(np.uint8)