from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, GaussianNoise, ReLU, LeakyReLU, Activation
from keras.layers import Concatenate, Add, UpSampling2D, AveragePooling2D, Lambda, BatchNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.constraints import Constraint, max_norm
from keras import initializers
from custom_layers import *

import math

# For Discriminator
def DC_block(x_shape, inp_channel, out_channel, name):

    assert x_shape[0] == x_shape[1]

    input_x = Input(shape=x_shape, name="Input-x")

    x = input_x

    x = LeakyReLU(0.2, name="LReLU1")(x)
    x = Conv2D(inp_channel, kernel_size=3, strides=1, padding="same", name="Conv1")(x)
    x = AveragePooling2D(name="DN-Sample")(x)
    x = LeakyReLU(0.2, name="LReLU2")(x)
    x = Conv2D(out_channel, kernel_size=3, strides=1, padding="same", name="Conv2")(x)

    x_shortcut = input_x

    if inp_channel != out_channel:
        x_shortcut = Conv2D(out_channel, kernel_size=1, strides=1, use_bias=False, name="ConvS")(x_shortcut)
    
    x_shortcut = AveragePooling2D(name="DN-SampleS")(x_shortcut)

    out = Add(name="ResdualAdd")([x, x_shortcut])
    out = Lambda(lambda x: x / math.sqrt(2.0))(out)

    return Model([input_x], [out], name=name)

# For the down-sampling of Generator and Style Encoder
def IN_block(x_shape, inp_channel, out_channel, name, res_add, dn_sample=False):

    assert x_shape[0] == x_shape[1]

    input_x = Input(shape=x_shape, name="Input-x")

    x = input_x
    x = InstanceNormalization(name="IN-1")(x)
    x = LeakyReLU(0.2, name="LReLU1")(x)
    x = Conv2D(inp_channel, kernel_size=3, strides=1, padding="same", kernel_initializer="he_uniform")(x)

    if dn_sample:
        x = AveragePooling2D(name="DN-Sample")(x)

    x = InstanceNormalization(name="IN-2")(x)
    x = LeakyReLU(0.2, name="LReLU2")(x)
    x = Conv2D(out_channel, kernel_size=3, strides=1, padding="same", kernel_initializer="he_uniform")(x)

    if res_add:
        x_shortcut = input_x

        if inp_channel != out_channel:
            x_shortcut = Conv2D(out_channel, kernel_size=1, strides=1, kernel_initializer="he_uniform", use_bias=False, name="ConvS")(x_shortcut)

        if dn_sample:
            x_shortcut = AveragePooling2D(name="DN-SampleS")(x_shortcut)

        out = Add(name="ResdualAdd")([x, x_shortcut])
        out = Lambda(lambda x: x / math.sqrt(2.0))(out)
    else:
        out = x

    return Model([input_x], [out], name=name)

# For the up-sampling of Generator
def AN_block(x_shape, s_shape, inp_channel, out_channel, name, c_shape=None, up_sample=False):

    assert x_shape[0] == x_shape[1]

    input_x = Input(shape=x_shape, name="Input-x")
    input_s = Input(shape=s_shape, name="Input-s")

    def style_affine(channel_size, name):
        block = Sequential(name=name)
        block.add(Dense(channel_size, kernel_initializer="he_uniform"))
        block.add(Reshape([1, 1, channel_size]))
        return block

    g1 = style_affine(inp_channel, name=f"StyleFC-G1")(input_s)
    g2 = style_affine(out_channel, name=f"StyleFC-G2")(input_s)

    x = input_x
    x = LeakyReLU(0.2, name="LReLU1")(x)
    x = SMod(name="S-Modulation-1")([x, g1]) # (x * (1 + g1))

    if up_sample:
        x = UpSampling2D(name="UP-Sample")(x)

    x = Conv2D(out_channel, kernel_size=3, strides=1, padding="same", kernel_initializer="he_uniform", use_bias=False)(x)
    x = CNom(name="C-Normalization-1")(x) # x / sigma
    x = BiasLayer(name="Bias-1")(x)
    
    x = LeakyReLU(0.2, name="LReLU2")(x)
    x = SMod(name="S-Modulation-2")([x, g2])
    x = Conv2D(out_channel, kernel_size=3, strides=1, padding="same", kernel_initializer="he_uniform", use_bias=False)(x)

    if c_shape:
        input_c = Input(shape=c_shape, name="Input-c")
        x = Add(name="ResdualAdd")([x, input_c])

    x = CNom(name="C-Normalization-2")(x)
    x = BiasLayer(name="Bias-2")(x)

    out = x

    if c_shape:
        return Model([input_x, input_s, input_c], [out], name=name)
    else:
        return Model([input_x, input_s], [out], name=name)

# For Region Detection branch of Generator
def SN_block(x_shape, m_shape, inp_channel, out_channel, name):

    assert x_shape[0] == x_shape[1]

    input_x = Input(shape=x_shape, name="Input-x")
    input_m = Input(shape=m_shape, name="Input-s")

    x = input_x

    x = BatchNormalization(name="BN-1")(x)
    x = ReLU(name="ReLU1")(x)
    x = Conv2D(out_channel, kernel_size=3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    
    x = UpSampling2D(name="UP-Sample")(x)

    x = BatchNormalization(name="BN-2")(x)
    x = ReLU(name="ReLU2")(x)
    x = Conv2D(out_channel, kernel_size=3, strides=1, padding="same", kernel_initializer="he_uniform")(x)

    out = Add(name="Skip-Connection")([x, input_m])

    return Model([input_x, input_m], [out], name=name)