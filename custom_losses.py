import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.losses import BinaryCrossentropy
from keras.models import Model, Sequential, load_model
from keras.layers.core import Lambda
from keras.layers import Reshape, Multiply
from PIL import Image
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

#path_to_weights_file=FILE_PATH+"/FAN/2DFAN-4_keras.h5"
def lm_loss(inp_face,out_face):
    
    lm = load_model("pretrained_lm/2DFAN-4_keras.h5")
    inp_face_t = tf.transpose(inp_face, [0, 3, 1, 2])
    lm_inp_t = lm(inp_face_t)
    out_face_t = tf.transpose(out_face, [0, 3, 1, 2])
    lm_out_t = lm(out_face_t)
    lm_inp = lm_inp_t[-1]    #(?,68,64,64)
    lm_out = lm_out_t[-1]
    
    q = K.square(lm_inp-lm_out)
    s1 = K.sum(q, axis = -1)
    print(s1.shape)
    s2 = K.sum(s1, axis = -1)
    print(s2.shape)
    m = K.mean(s2, axis = -1)
    return m
    
    
    #model_lm = tf.keras.models.load_model("pretrained_lm/2DFAN-4_keras.h5")
    
    #lm_model = tf.saved_model.load_v2("pretrained_wingloss", tags=None)
    #lm = lm_model.signatures["serving_default"]
    #print(lm)  # tensorflow.python.saved_model.load._WrapperFunction object at 0x7f758cbb43c8>

    #print(lm.input)
    #i = lm(inp_face)['landmark']
    #o = lm(out_face)['landmark']
    #print(i.shape) # (?, 136)
    #print(i)       # Tensor("loss_2/cropping2d_2_loss/lm_loss/StatefulPartitionedCall:2", shape=(?, 136), dtype=float32)

    #s = K.square(i-o)
    #m = K.mean(s, axis = -1)
    #print(m.shape) # (?,)
    #return m
    
    #lm = Model(inputs=model_lm.input, outputs=model_lm.output)
    #a = np.ones((8,3,256,256))
    #lll = lm.predict(a, steps=1)
    #print(lll)
    #return 0
    #print(inp_face)#Tensor("G_Model_target_1:0", shape=(?, ?, ?, ?), dtype=float32)

    #print(out_face)#Tensor("G_Model/To-Image/conv2d_1/BiasAdd:0", shape=(?, 256, 256, 3), dtype=float32, device=/device:GPU:0)

    #inp_face_t = tf.transpose(inp_face, [0, 3, 1, 2])
        #print("inp not None")
    #lm_inp_t = lm(inp_face_t)
        #print(pred_inp.shape)
    
    #out_face_t = tf.transpose(out_face, [0, 3, 1, 2])
        #print("out not None")
    #print(out_face_t) # Tensor("loss_1/G_Model_loss_1/lm_loss/transpose:0", shape=(?, 3, 256, 256), dtype=float32)

    #lm_out_t = lm(out_face_t)
    #print(pred_out) # list object contains 4 tensors, each tensor shape is (?,68,64,64)
    #print(lm_out)
    #lm_inp = lm_inp_t[-1]
   # lm_out = lm_out_t[-1]
    #print(lm_inp) #Tensor("loss_1/G_Model_loss_1/lm_loss/model_1/l30.6457242737270144/BiasAdd:0", shape=(?, 68, ?, ?), dtype=float32)

    
    
    
    """
    lm_inp_flat = K.reshape(lm_inp, (8, 68, 64*64))  #(?,68,64*64)
    idx_inp = K.argmax(lm_inp_flat, axis=-1)           #(?, 68)
    #idx_inp_reshape = K.reshape(idx_inp, (8, 68, 1))   # (?, 68, 1)
    #pred_inp = K.repeat_elements(idx_inp_reshape, 2,2)                                       # (?, 68, 2)
    #pred_inp[:,:,0] = pred_inp[:,:,0]% 64
    #pred_inp[:,:,1] = pred_inp[:,:,1]/ 64
    
    
    lm_out_flat = K.reshape(lm_out, (8, 68, 64*64))  #(?,68,64*64)
    idx_out = K.argmax(lm_out_flat, axis=-1)           #(?, 68)
    #idx_out_reshape = K.reshape(idx_out, (8, 68, 1))   # (?, 68, 1)
    #pred_out = K.repeat_elements(idx_out_reshape, 2,2)                                       # (?, 68, 2)
    #pred_out[:,:,0] = pred_out[:,:,0]% 64
    #pred_out[:,:,1] = pred_out[:,:,1]/ 64
    
    
    
    q = K.square((idx_inp % 64) - (idx_out % 64))
    qq = K.square((idx_inp / 64) - (idx_out / 64))
    q = K.cast(q, dtype='float32')
    qq = K.cast(qq, dtype='float32')
    #print(q.shape) # (?,68)
    loss = K.mean(q+qq, axis = -1)
    #print(loss) # (?, )
    return loss
    
    #lm_model = tf.saved_model.load_v2("pretrained_lm", tags=None)
    #lm = lm_model.signatures["serving_default"]
    #mm = Model(inputs=lm.input, outputs=lm.output)
    
    # inp
    #inp_batch_inds = lm(inp_face)['landmark']
        
    #lm_model = tf.keras.models.load_model("pretrained_lm")
    #lm = lm_model.signatures["serving_default"]
    """
    
    """
    inp_batch_inds = mm(inp_face)['landmark']
    
    inp_batch_inds = tf.reshape(inp_batch_inds, [-1, 68, 2])
    inp_batch_inds = 160*inp_batch_inds
    inp_land = inp_batch_inds[:, 17:, :]
    
    # out
    out_batch_inds = lm(out_face)['landmark']
    out_batch_inds = tf.reshape(out_batch_inds, [-1, 68, 2])
    out_batch_inds = 160*out_batch_inds
    out_land = out_batch_inds[:, 17:, :]
    
    # loss
    q = K.square(inp_land - out_land)
    print(q.shape)
    #s = K.sum(q, axis = (1,2))
    #print(s.shape)
    
    landmarks_loss = K.mean(K.mean(q, axis=-1), axis = -1)
    print(landmarks_loss.shape)
    return landmarks_loss  
    """
def ID_loss(inp_face,out_face):
    #print(inp_face.shape)
    #print(out_face.shape)
    vggface = VGGFace(model='senet50', include_top=False, pooling='avg')#, input_shape=(224, 224, 3)
    model_vgg = Model(inputs=vggface.input, outputs=vggface.output)
    
    
    inp_emb = model_vgg(inp_face)
    out_emb = model_vgg(out_face)  #shape=(?,2048)
    
    a = K.sqrt(K.sum(K.square(out_emb - inp_emb), axis=-1))# Euclidean distance
    #a = K.clip(a,85,150)
    return a  # shape = (?)
    
def BCE_Loss(Y_true, Y_pred):
    #print("This is BCE loss")
    #print(Y_true.shape)   # shape=(?,?)
    #print(Y_pred.shape)   # shape=(?,1)
    b = K.mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_true, logits=Y_pred), axis=-1)
    #print(b.shape)        # shape=(?,)
    return b

def reconstruction_loss(Y_true, Y_pred):
    #print("This is reconstruction_loss")
    #print(Y_true.shape)   # shape=(?,?,?,?)
    #print(Y_pred.shape)   # shape=(?,256,256,3)
    c = K.mean(K.abs(Y_true - Y_pred), axis=-1)
    #print(c.shape)        # shape=(?,256,256)
    return c

def gradient_l2_penalty_loss(x_hat, d_hat):
    gradients = K.gradients(d_hat, x_hat)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)

def gradient_l1_penalty_loss(x_hat, d_hat):
    gradients = K.gradients(d_hat, x_hat)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    return 0.5 * K.mean(gradients_sqr_sum, axis=0)


