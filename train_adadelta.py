import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)
n_epochs=10
batch_size = 28
LR = 0.001              # learning rate

mnist = input_data.read_data_sets('./data/', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images
test_y = mnist.test.labels

# plot one example
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

print(mnist.test.images.shape)
print(mnist.test.labels.shape)

tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
image = tf.reshape(tf_x, [-1, 28, 28, 1])
tf_y = tf.placeholder(tf.int32, [None, 10])


conv1 = tf.layers.conv2d(inputs=image, filters=64, kernel_size=3, strides=2, padding='valid',  activation=tf.nn.relu)
conv2 = tf.layers.conv2d(conv1, 32, 2, 1, 'same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv2, 2, 1)  
pool1 = tf.layers.dropout(pool1, rate=0.35, training=True)
fc1 = tf.contrib.layers.flatten(pool1)
fc1 = tf.layers.dense(fc1, 256)
fc1 = tf.layers.dropout(fc1, rate=0.5, training=True)
output = tf.layers.dense(fc1, 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdadeltaOptimizer(LR).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op) 



train_cost_summary = tf.summary.scalar("train_loss", loss)
train_acc_summary = tf.summary.scalar("train_accuracy", accuracy)
test_cost_summary = tf.summary.scalar("test_loss", loss)
test_acc_summary = tf.summary.scalar("test_accuracy", accuracy)


writerVal = tf.summary.FileWriter('./Val', graph=tf.get_default_graph())
writerTrain = tf.summary.FileWriter('./train', graph=tf.get_default_graph())

trainLoss=range(0,n_epochs)
testLoss=range(0,n_epochs)
trainacc=range(0,n_epochs)
testacc=range(0,n_epochs)

for epoch in range(n_epochs):
    train_loss = 0.0;
    train_acc=0.0;
    batch_count = int(mnist.train.num_examples/batch_size)
    for i in range(batch_count):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, trainb_cost, trainb_acc, _train_cost_summary, _train_acc_summary =  sess.run([train_op, loss, accuracy, train_cost_summary, train_acc_summary],feed_dict={tf_x: batch_x, tf_y: batch_y})
        train_loss=train_loss+(trainb_cost/batch_count)
        train_acc=train_acc+(trainb_acc/batch_count)
    
    writerTrain.add_summary(_train_cost_summary, epoch)
    writerTrain.add_summary(_train_acc_summary, epoch)
    writerTrain.flush()
    test_loss, test_acc, _test_cost_summary, _test_acc_summary =  sess.run([loss, accuracy, test_cost_summary, test_acc_summary], feed_dict={tf_x: mnist.test.images, tf_y: mnist.test.labels})
               
    writerVal.add_summary(_test_cost_summary, epoch)
    writerVal.add_summary(_test_acc_summary, epoch)
    writerVal.flush()
    trainLoss[epoch]=train_loss
    testLoss[epoch]=test_loss
    trainacc[epoch]=train_acc
    testacc[epoch]=test_acc
            
    print('Epoch {0:3d} | Train Loss: {1:.5f} | Test Loss: {2:.5f} | Train acc: {3:.5f} | Test acc: {4:.5f}' .format(epoch, train_loss, test_loss, train_acc, test_acc))        

epoch_count = range(0, n_epochs)
# Visualize loss history
plt.figure(1)
plt.plot(epoch_count, trainLoss, 'r--')
plt.plot(epoch_count, testLoss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Cross entropy Loss')
plt.title('Loss Vs Epoch (Adadelta optimizer)')
#plt.show();
plt.savefig('Adadelta_cross_loss.jpg')

plt.figure(2)	
plt.plot(epoch_count, trainacc, 'r--')
plt.plot(epoch_count, testacc, 'b-')
plt.legend(['Training acc', 'Test acc'])
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.title('Acc Vs Epoch (Adadelta optimizer)')
plt.savefig('Adadelta_acc.jpg')
#plt.show(); 
  
