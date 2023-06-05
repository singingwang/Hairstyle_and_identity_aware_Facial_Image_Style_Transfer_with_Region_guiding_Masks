import os
from keras import backend as K
from keras.models import load_model

class ExponentialMovingAverage:

    def __init__(self, models, decay=0.999):
        self.decay = decay
        self.model_weights = [(model, K.batch_get_value(model.weights)) for model in models]

    def update(self):
        for (model, ema_weights) in self.model_weights:
            for i, (w1, w2) in enumerate(zip(ema_weights, K.batch_get_value(model.weights))):
                ema_weights[i] = w1 - ((1 - self.decay) * (w1 - w2))

    def EMA_load(self, directory, paths):

        old_weights_list = []
        tmp_models = []

        for (model, ema_weights), path in zip(self.model_weights, paths):
            old_weights_list.append(K.batch_get_value(model.weights))
            model.load_weights(os.path.join(directory, path))
            tmp_models.append(model)

        self.model_weights = [(model, K.batch_get_value(model.weights)) for model in tmp_models]

        for (model, ema_weights), old_weights in zip(self.model_weights, old_weights_list):
            K.batch_set_value(zip(model.weights, old_weights))

    def EMA_save(self, directory, paths):
        
        for (model, ema_weights), path in zip(self.model_weights, paths):

            old_weights = K.batch_get_value(model.weights)

            K.batch_set_value(zip(model.weights, ema_weights))
            model.save_weights(os.path.join(directory, path))
            K.batch_set_value(zip(model.weights, old_weights))

    def EMA_test(self, test_func):

        old_weights_list = []

        for (model, ema_weights) in self.model_weights:
            old_weights_list.append(K.batch_get_value(model.weights))
            K.batch_set_value(zip(model.weights, ema_weights))

        test_func()

        for (model, ema_weights), old_weights in zip(self.model_weights, old_weights_list):
            K.batch_set_value(zip(model.weights, old_weights))
