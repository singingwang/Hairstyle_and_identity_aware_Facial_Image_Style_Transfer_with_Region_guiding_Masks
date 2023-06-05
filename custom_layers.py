import tensorflow as tf
from tensorflow import pad

from keras import backend as K
from keras.utils import conv_utils
from keras.layers import Layer, Reshape, Add, MaxPooling2D
from keras.layers import Conv2D, Dense, Conv2DTranspose

from keras import activations, initializers, regularizers, constraints
from keras.engine.base_layer import InputSpec
from keras.constraints import Constraint
import numpy as np
from functools import partial

class MinPooling2D(MaxPooling2D):
	
	def __init__(self, **kwargs):
		super(MinPooling2D, self).__init__(**kwargs)
	
	def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
		return -K.pool2d(-inputs, pool_size, strides, padding, data_format, pool_mode="max")

class EqualizedDense(Dense):

	def __init__(self, units, gain=np.sqrt(2), lrmul=1.0, **kwargs):
		self.gain = gain
		self.lrmul = lrmul
		super(EqualizedDense, self).__init__(units=units, kernel_initializer=initializers.random_normal(stddev=1.0/lrmul), bias_initializer="zeros", **kwargs)

	def build(self, input_shape):
		super(EqualizedDense, self).build(input_shape)
		fan_in = np.prod(self.kernel.shape[:-1]).value
		self.wscale = (self.gain / np.sqrt(fan_in)) * self.lrmul
		print(self.name, self.wscale)

	def call(self, inputs):

		output = K.dot(inputs, self.kernel * self.wscale)

		if self.use_bias:
			output = K.bias_add(output, self.bias, data_format='channels_last')

		if self.activation is not None:
			output = self.activation(output)

		return output

class EqualizedConv2D(Conv2D):

	def __init__(self, filters, kernel_size, gain=np.sqrt(2), lrmul=1.0, **kwargs):
		self.gain = gain
		self.lrmul = lrmul
		super(EqualizedConv2D, self).__init__(filters=filters, kernel_size=kernel_size, kernel_initializer=initializers.random_normal(stddev=1.0/lrmul), bias_initializer="zeros", **kwargs)

	def build(self, input_shape):
		super(EqualizedConv2D, self).build(input_shape)
		fan_in = np.prod(self.kernel.shape[:-1]).value
		self.wscale = (self.gain / np.sqrt(fan_in)) * self.lrmul

	def call(self, inputs):
		outputs = K.conv2d(inputs, self.kernel * self.wscale, strides=self.strides, padding=self.padding, data_format=self.data_format, dilation_rate=self.dilation_rate)

		if self.use_bias:
			outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

		if self.activation is not None:
			return self.activation(outputs)
		else:
			return outputs

class EqualizedConv2DTranspose(Conv2DTranspose):

	def __init__(self, filters, kernel_size, gain=np.sqrt(2), lrmul=1.0, **kwargs):
		self.gain = gain
		self.lrmul = lrmul
		super(EqualizedConv2DTranspose, self).__init__(filters=filters, kernel_size=kernel_size, kernel_initializer=initializers.random_normal(stddev=1.0/lrmul), bias_initializer="zeros", **kwargs)

	def build(self, input_shape):
		super(EqualizedConv2DTranspose, self).build(input_shape)
		fan_in = np.prod(self.kernel.shape[:-1]).value
		self.wscale = (self.gain / np.sqrt(fan_in)) * self.lrmul
		print(self.name, self.wscale)

	def call(self, inputs):
		input_shape = K.shape(inputs)
		batch_size = input_shape[0]

		if self.data_format == 'channels_first':
			h_axis, w_axis = 2, 3
		else:
			h_axis, w_axis = 1, 2

		height, width = input_shape[h_axis], input_shape[w_axis]
		kernel_h, kernel_w = self.kernel_size
		stride_h, stride_w = self.strides

		if self.output_padding is None:
			out_pad_h = out_pad_w = None
		else:
			out_pad_h, out_pad_w = self.output_padding

		# Infer the dynamic output shape:
		out_height = conv_utils.deconv_length(height, stride_h, kernel_h, self.padding, out_pad_h, self.dilation_rate[0])
		out_width = conv_utils.deconv_length(width, stride_w, kernel_w, self.padding, out_pad_w, self.dilation_rate[1])
		
		if self.data_format == 'channels_first':
			output_shape = (batch_size, self.filters, out_height, out_width)
		else:
			output_shape = (batch_size, out_height, out_width, self.filters)

		outputs = K.conv2d_transpose(inputs, self.kernel * self.wscale, output_shape, self.strides, padding=self.padding, data_format=self.data_format, dilation_rate=self.dilation_rate)

		if self.use_bias:
			outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

		if self.activation is not None:
			return self.activation(outputs)

		return outputs

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'REFLECT')

class AdaIN(Layer):

    def __init__(self, **kwargs):
        super(AdaIN, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]

    def call(self, x):
        assert isinstance(x, list)
        content, gamma, beta = x[0], x[1], x[2]

        epsilon = 1e-5

        meanC, varC = tf.nn.moments(content, [1, 2], keep_dims=True)
        sigmaC = tf.sqrt(tf.add(varC, epsilon))
    
        return (1 + gamma) * ((content - meanC) / sigmaC) + beta

class CNom(Layer):
	
	def __init__(self, center=False, **kwargs):
		super(CNom, self).__init__(**kwargs)
		self.center = center

	def call(self, x):
		epsilon = 1e-5

		meanC, varC = tf.nn.moments(x, [1, 2], keep_dims=True)
		sigmaC = tf.sqrt(tf.add(varC, epsilon))
		
		if self.center:
			return (x - meanC) / sigmaC
		else:
			return x / sigmaC

class CMod(Layer):

	def __init__(self, *args, **kwargs):
		super(CMod, self).__init__(**kwargs)

	def build(self, input_shape):
		channel_size = input_shape[-1]
		self.gamma = self.add_weight("gamma", shape=(1, 1, channel_size), initializer="zeros", trainable=True)

	def call(self, x):
		return (1 + self.gamma) * x

class SMod(Layer):
	
	def __init__(self, center=False, **kwargs):
		super(SMod, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		assert isinstance(input_shape, list)
		return input_shape[0]

	def call(self, inputs):
		assert isinstance(inputs, list)
		x, gamma = inputs
		return (1 + gamma) * x
		
class BiasLayer(Layer):
	def __init__(self, *args, **kwargs):
		super(BiasLayer, self).__init__(*args, **kwargs)
	
	def build(self, input_shape):
		channel_size = input_shape[-1]
		self.bias = self.add_weight("bias", shape=(channel_size,), initializer="zeros", trainable=True)

	def call(self, x):
		return K.bias_add(x, self.bias)

class NoiseLayer(Layer):

	def __init__(self, strength=1.0, *args, **kwargs):
		super(NoiseLayer, self).__init__(*args, **kwargs)
		self.strength = K.variable(strength, name="strength")

	def build(self, input_shape):
		self.weight = self.add_weight("noise-weight", shape=(input_shape[-1], ), initializer="he_uniform", trainable=True)
		
	def call(self, x):
		noise = K.random_normal(shape=(x.shape[1], x.shape[2], 1))
		return x + noise * self.weight * self.strength

class WeightedSum(Add):

	def __init__(self, alpha=1.0, **kwargs):
		super(WeightedSum, self).__init__(**kwargs)
		self.alpha = K.variable(alpha, name='ws_alpha')
 
	def _merge_function(self, inputs):
		assert (len(inputs) == 2)
		output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
		return output

# pixel-wise feature vector normalization layer
class PixelNormalization(Layer):
	# initialize the layer
	def __init__(self, **kwargs):
		super(PixelNormalization, self).__init__(**kwargs)
 
	# perform the operation
	def call(self, inputs):
		values = inputs**2.0
		mean_values = K.mean(values, axis=-1, keepdims=True)
		mean_values += 1e-8
		l2 = K.sqrt(mean_values)
		normalized = inputs / l2
		return normalized
 
	# define the output shape of the layer
	def compute_output_shape(self, input_shape):
		return input_shape

class MinibatchStdev(Layer):
	# initialize the layer
	def __init__(self, **kwargs):
		super(MinibatchStdev, self).__init__(**kwargs)
 
	# perform the operation
	def call(self, inputs):
		# calculate the mean value for each pixel across channels
		mean = K.mean(inputs, axis=0, keepdims=True)
		# calculate the squared differences between pixel values and mean
		squ_diffs = K.square(inputs - mean)
		# calculate the average of the squared differences (variance)
		mean_sq_diff = K.mean(squ_diffs, axis=0, keepdims=True)
		# add a small value to avoid a blow-up when we calculate stdev
		mean_sq_diff += 1e-8
		# square root of the variance (stdev)
		stdev = K.sqrt(mean_sq_diff)
		# calculate the mean standard deviation across each pixel coord
		mean_pix = K.mean(stdev, keepdims=True)
		# scale this up to be the size of one input feature map for each sample
		shape = K.shape(inputs)
		output = K.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
		# concatenate with the output
		combined = K.concatenate([inputs, output], axis=-1)
		return combined
 
	def compute_output_shape(self, input_shape):
		# create a copy of the input shape as a list
		input_shape = list(input_shape)
		# add one to the channel dimension (assume channels-last)
		input_shape[-1] += 1
		# convert list to a tuple
		return tuple(input_shape)