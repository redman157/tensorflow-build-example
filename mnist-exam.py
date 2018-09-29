import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import datasets
mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
class_name =['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
def loss(ligits,labels):
  return tf.reduce_mean(-tf.reduce_sum(tf.matmul(ligits,tf.log(labels),tf.matmul(1-ligits,tf.log(1-labels)))))

def optimize(loss,learning_rate):
  return tf.train.AdamOptimizer(learning_rate).minimize(loss)

def computed_accuracy(labels,ligits):
  correct_predictions = tf.argmax(ligits, axis = 1,dtype= tf.int64)
  labels = tf.cast(labels,tf.int64)
  batch_size = int(logits.shape[0])
  return tf.reduce_sum(tf.cast(tf.equal(correct_predictions,labels),tf.float32))/batch_size

def model(X,w,b):
  w = tf.Variable(tf.float32,shape=[1,1])
  b = tf.Variable(tf.float32,shape=[1,1])
  return tf.add(tf.matmul(X,w),b)

def step_counter(optimize):
  couter = 0
  for i in range(batch_size):
    optimize(loss,learning_rate)
    counter = counter +1
  return counter

def train(train_images,train_labels, epoch = 10):
  

def test(test_images,test_labels, epoch = 10):
  pass

def plot_image(i,predictions_array,true_label,ima):
  predictions_array,true_label,img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

def plot_value_array
  
      
