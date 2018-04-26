import numpy as np
import pandas as pd
import tensorflow as tf


# read data and split into training and development datasets
# Features and labels were merged into the train, development and test datasets
train_df = pd.read_csv('/Users/arjunsatheesan/Downloads/competition-datasets/projectgroup7_mushroom-edibility-dataset/train/train.csv', header=None)
dev_df = pd.read_csv('//Users/arjunsatheesan/Downloads/competition-datasets/projectgroup7_mushroom-edibility-dataset/development/development.csv', header=None)
test_df= pd.read_csv("/Users/arjunsatheesan/Downloads/competition-datasets-test-no-test-y/projectgroup7_mushroom-edibility-dataset/test/test.csv",header=None)

# split the train and dev datasets into their corresponding features and labels
train_labels = train_df.iloc[:,0].as_matrix()
train_features = train_df.iloc[:,1:].as_matrix()

dev_labels = dev_df.iloc[:,0].as_matrix()
dev_features = dev_df.iloc[:,1:].as_matrix()

test_features = test_df.as_matrix()

# parameters
epochs= 100000
batch_size=250
display_step = 1000
feature_size = train_features.shape[1]
class_size=1
epoch_index=0
epochs_completed=0


# Computation graph construction
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))

def next_batch(batch_size):

    global train_features
    global train_labels
    global epoch_index
    global epochs_completed

    start = epoch_index
    epoch_index += batch_size

    if epoch_index > len(train_features):
        epochs_completed += 1
        p = np.random.permutation(len(train_features))
        train_features = train_features[p]
        train_labels = train_labels[p]
# start next epoch
        start = 0
        epoch_index = batch_size
        assert batch_size <= len(train_features)
    end = epoch_index
    return train_features[start:end], train_labels[start:end]

def compute_accuracy(features, labels):
    global prediction
    y_prediction = sess.run(prediction, feed_dict={x: features})
    y_prediction = tf.greater(y_prediction,0.5)
    correct_prediction = tf.equal(y_prediction, tf.equal(y,1.0))
    accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )
    result = sess.run(accuracy, feed_dict={x: features, y: labels})
    return result

def compute_results(features):
    global prediction
    y_prediction = sess.run(prediction, feed_dict={x: features})
    y_prediction = tf.cast(tf.greater(y_prediction,0.5),tf.int32)
    return y_prediction

# change the format of the labels according to one-hot encoding

def one_hot_encode(labels, classes=10):
    one_hot = np.zeros([len(labels), 2])
    for i in range(len(labels)):
        one_hot[i, labels[i]] = 1.
    return one_hot


train_labels = one_hot_encode(train_labels)
dev_labels = one_hot_encode(dev_labels)

# create the computational graph
x = tf.placeholder(tf.float32,[None, feature_size])
y = tf.placeholder(tf.float32,[None,2])

# Hidden layer
W1 = weight_variable([feature_size,50])
b1 = bias_variable([50])
hidden_output = tf.add(tf.matmul(x,W1),b1)
hidden_output = tf.nn.relu(hidden_output)


# Output layer
W2 = weight_variable([50, 2])
b2 = bias_variable([2])

output = tf.add(tf.matmul(hidden_output,W2),b2)
prediction = tf.nn.sigmoid(output)

# Loss function and optimizer
# Here, the Gradient Descent optimizer performs back propagation as well
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,labels=y))
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# Initialize all the variables
init = tf.global_variables_initializer()

# Run session for the computation graph
sess = tf.Session()
sess.run(init)

for i in range(epochs):
    batch_features, batch_labels = next_batch(batch_size)

    sess.run(optimizer, feed_dict={x: batch_features, y: batch_labels})
    if i%display_step ==0:
        print("Training accuracy for epoch#:{} ".format(i),compute_accuracy(batch_features,batch_labels))

print("Development accuracy: ",compute_accuracy(dev_features,dev_labels))
test_result = sess.run(prediction, feed_dict={x: test_features})
for element in test_result:
    print(np.argmax(element))
