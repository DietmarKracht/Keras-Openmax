from typing import Dict, Any, Union

import numpy as np
from keras import backend as K
import scipy.spatial.distance as spd
import os

from numpy.core._multiarray_umath import ndarray
from tensorflow.keras.models import load_model
from EVT_fitting import weibull_tailfitting
from compute_openmax import recalibrate_scores
from tqdm import tqdm


class Openmax:
    """Class for openmax computations.
    Constructor needs a pretrained model with 2 named layers. One should represent the activation vectors and the other
    contains the softmax scores.  The input data and labels as a tuple, the input shape of the first layer,
    optional arguments that represent the observed named layers of the model(eg. dense_01, FNC7, softmax...)
    The layer names have to match the layer names of the model
    """
    DATA_PATH = "data/openmax/"
    MODEL_PATH = "model/openmax/"

    def __init__(self, model, x_data, y_data, input_shape: tuple, name: str, overwrite=False, *args):
        self.threshold = 0.5
        self.weibull_model = None
        self.input_shape = (1,) + input_shape
        self.model = model
        self.x_data = x_data
        self.y_data = y_data
        self.name = name
        self.observations = []
        self.backend_functions = []
        self.CLASSES = np.unique(self.y_data)
        self.overwrite = overwrite
        for observation in args:
            self.observations.append(observation)
            self.init_backend_functions(observation)
        if not os.path.exists(self.DATA_PATH+self.name+"/"):
            os.makedirs(self.DATA_PATH+self.name+"/")
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(self.MODEL_PATH)

    def save_model(self):
        self.model.save(self.MODEL_PATH + self.name+".h5")

    def load_model(self, custom_objects=None):
        self.model = load_model(self.MODEL_PATH + self.name+".h5", custom_objects=custom_objects)
        return self.model

    def get_correctly_classified(self):
        predictions = self.model.predict(self.x_data)
        out = []
        for idx, p in enumerate(predictions):
            p_label = np.argmax(p)
            if p_label == self.y_data[idx] and np.amax(p) > self.threshold:
                out.append(True)
            else:
                out.append(False)
        percentage = sum(out)/len(self.x_data)
        print("{} percent of the data were predicted correctly and are considered for the probability distribution.".format(percentage*100))
        return np.array(out)

    def init_backend_functions(self, layer):
        if type(layer) is str:
            self.backend_functions.append(K.function([self.model.layers[0].input, K.learning_phase()],
                                         [self.model.get_layer(layer).output]))
        else:
            self.backend_functions.append(K.function([self.model.layers[0].input, K.learning_phase()],
                                         [self.model.layers[layer].output]))

    def get_backend_function(self, layer, input):
        return self.backend_functions[self.observations.index(layer)]([input, 0])[0]

    def compute_feature(self, image_set):
        if image_set.shape == self.input_shape:
            return np.asarray(self.get_backend_function(self.observations[0], image_set))
        out = []
        for image in image_set:
            out.append(self.get_backend_function(self.observations[0], image.reshape(self.input_shape)))
        return np.asarray(out)

    def compute_score(self, image_set):

        if image_set.shape == self.input_shape:
            return np.asarray(self.get_backend_function(self.observations[1], image_set))
        out = []
        for image in image_set:
            out.append(self.get_backend_function(self.observations[1], image.reshape(self.input_shape)))
        return np.asarray(out)

    def compute_means(self, feature):
        return np.mean(feature, axis=0)

    def compute_distance(self, mean_train_vector, feature, category_name):
        eu_dist, cos_dist, eucos_dist = [], [], []
        for feat in feature:
            eu_dist += [spd.euclidean(mean_train_vector, feat)]
            cos_dist += [spd.cosine(mean_train_vector, feat)]
            eucos_dist += [spd.euclidean(mean_train_vector, feat) / 200. +
                           spd.cosine(mean_train_vector, feat)]
        distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean': eu_dist}
        return distances

    def prepare_data(self, x, y):
        index = np.argsort(y)
        return x[index], y[index]

    def sort_data(self):
        index = self.get_correctly_classified()
        x_data = self.x_data[index]
        y_data = self.y_data[index]
        sorted_x, sorted_y = self.prepare_data(x_data, y_data)
        return sorted_x, sorted_y

    def compute_data(self):
        print("Started Data Computation")
        x_data, y_data = self.sort_data()
        all_means = []
        all_distances = []
        for i in tqdm(range(len(self.CLASSES))):
            feat = self.compute_feature(x_data[y_data == i])
            mean = self.compute_means(feat)
            distance = self.compute_distance(mean, feat, i)
            all_means.append(mean)
            all_distances.append(distance)
            np.save(self.DATA_PATH + self.name + "/" + 'activation_vector_class' + str(i), feat)
        np.save(self.DATA_PATH + self.name + "/" + 'mean', all_means)
        np.save(self.DATA_PATH + self.name + "/" + 'distance', all_distances)
        print("Finished Data Computation")

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def create_weibull(self, distance_type: str, tailsize=20, overwrite=False):
        if overwrite:
            self.compute_data()
        elif len(os.listdir(self.DATA_PATH + self.name+"/")) == 0:
            self.compute_data()
        means = np.load(self.DATA_PATH + self.name + "/" + 'mean.npy', allow_pickle=True)
        distances = np.load(self.DATA_PATH + self.name + "/" + 'distance.npy', allow_pickle=True)
        self.weibull_model = weibull_tailfitting(means, distances, range(len(self.CLASSES)),
                                                 distance_type=distance_type, tailsize=tailsize)
        return self.weibull_model

    def predict(self, image, distance_type='eucos', alpha=10):
        if self.weibull_model is None:
            self.create_weibull(distance_type, self.overwrite)
        if not image.shape == self.input_shape:
            image = image.reshape(self.input_shape)
        imgarr = {'scores': self.compute_score(image),
                      self.observations[0]: self.compute_feature(image)}
        openmax, softmax = recalibrate_scores(self.weibull_model, range(len(self.CLASSES)), imgarr,
                                              self.observations[0], distance_type=distance_type,
                                              classes=len(self.CLASSES),alpharank=alpha)
        return openmax, softmax

    def adapt_tailsize(self, distance_type='eucos', tail_range=range(0,40), alpha_range=range(0,10)):
        adaption = np.zeros((len(alpha_range) + 1, len(tail_range) + 1))
        for image in tqdm(self.x_data):
            for alpha in alpha_range:
                for tail in tail_range:
                    self.create_weibull(distance_type, tailsize=tail, overwrite=False)
                    if not image.shape == self.input_shape:
                        image = image.reshape(self.input_shape)
                    imgarr = {'scores': self.compute_score(image),
                              self.observations[0]: self.compute_feature(image)}
                    openmax, softmax = recalibrate_scores(self.weibull_model, range(len(self.CLASSES)), imgarr,
                                                          self.observations[0], distance_type=distance_type,
                                                          classes=len(self.CLASSES), alpharank=alpha)
                    if np.argmax(openmax) == np.argmax(softmax):
                        adaption[alpha][tail] += 1
        return adaption
