import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimisers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape([len(x_train), 784, ])
x_test = x_test.reshape([len(x_test), 784, ])
x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# NAM feature network hyperparameters
batch_size = 1024
epochs = 1000
alpha_anneal = 0.995

alpha = 0.001
weight_decay_coeff = 0.000001
feat_dropout = 0.05

alpha_scheduler = LearningRateSchedule(alpha, )
optim = Adam(learning_rate=alpha)

# DNN hyperparameters
dropout_coeff = 0.05
dnn_weight_decay_coeff = 0.0000001
dnn_alpha = 0.001

# NAM feature network
class FeatureNetwork:
	def __init__(self):
		self.model = Sequential()
		
	def get_alpha(self, epoch, alpha):
		return alpha
		
	def get_feat_network(self, input_dim):
		self.model.add(Dense(64, input_shape=input_dim, activation="relu"))
		self.model.add(Dense(64, input_shape=input_dim, activation="relu"))
		self.model.add(Dense(32, input_shape=input_dim, activation="relu"))

		# using BCE instead of MSE because it's a classification problem, not regression
		self.model.compile(optimiser=optim, loss="binary_crossentropy", metric=["mse"])
		
		return self.model

class DNNLayer:
	def __init__(self):
		self.layer = []
		
	def get_layer(self, units):
		for _ in range(units):
			feat_neuron = FeatureNetwork()
			self.layer.append(feat_neuron)
			
		return self.layer
		
class DNN:
	def __init__(self):
		self.model = []
		
	def configure_model(self):
		l1 = DNNLayer()
		l2 = DNNLayer()
		l3 = DNNLayer()
		
		l1_layer = l1.get_layer(64)
		l2_layer = l1.get_layer(64)
		l3_layer = l1.get_layer(32)
		
		self.model.append(l1)
		self.model.append(l2)
		self.model.append(l3)
		
		return self.model
		
dnn = DNN()