from PIL import Image
import numpy as np

train_set='train.txt'
val_set='val.txt'
global pointer
pointer= 0
global train_lines
global val_lines
train_lines= open(train_set,'r').readlines()
val_lines= open(val_set,'r').readlines()
import tensorflow as tf
sess = tf.InteractiveSession()




def getlines(num,test=False):
    imgs = []
    labels = []
    global pointer
    if test:
        file_lines=val_lines
        pointer=0
    else:
        file_lines=train_lines

    for i in range(num):
        pointer+=1
        if pointer>=file_lines.__len__(): pointer=0
        #if test: print pointer
        line=file_lines[pointer]
        #print 'ImgSort/'+line.split(' ')[0]
        img=Image.open('ImgSort/'+line.split(' ')[0])
        img= img.convert("L").resize((28,28))
        #imgs.append()

        label =[0,0,0 , 0,0,0 , 0,0,0, 0]
        index = int(line.split(' ')[1])
        label[index]=1

        if(i==0):
            img=np.reshape(np.array(img),784)
            imgs=np.expand_dims(img,axis=0)
            #print imgs.shape
            labels=np.expand_dims(np.array(label),axis=0)
        else:
            img=np.reshape(np.array(img),784)
            img = np.expand_dims(img,axis=0)
            imgs = np.append(imgs, img, axis=0)
            label=np.expand_dims(np.array(label),axis=0)
            labels=np.append(labels,label,axis=0)
        #print imgs.shape
        #print labels.shape

    return imgs,labels

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())
#y = tf.matmul(x,W) + b



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def norm(x):
    return tf.nn.local_response_normalization(x)

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])
# this is the input tensor, 28*28 with only one channel
# the number of the picture( the first dimension) doesn't matter
h_conv1 = tf.nn.relu(norm(conv2d(x_image, W_conv1) + b_conv1))
# W_conv1 is an array of kernels, the size is 5*5*1 and and number is 32
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
# W_conv2 is another array of kernels, the size is 5*5*32 and the number is 64
# because the last layer make 32 kernels, so this dimension has the lenth of 32
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(norm(conv2d(h_pool1, W_conv2) + b_conv2))
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 64])
# W_conv2 is another array of kernels, the size is 5*5*32 and the number is 64
# because the last layer make 32 kernels, so this dimension has the lenth of 32
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(norm(conv2d(h_pool2, W_conv3) + b_conv3))
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([4 * 4 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess, './model-9999')

#load the model


#print sess.run(y_conv,feed_dict={x: getlines(1,True)[0]})
def predict(img):
    # img = img.convert("L").resize((28, 28))
    # img = np.reshape(np.array(img), 784)
    # img = np.expand_dims(img, axis=0)
    print img.shape
    result= sess.run(y_conv,feed_dict={x:img , keep_prob: 1.0})
    print result.shape
    answer=result.argmax()
    return answer

#load your picture and you will get your result
print predict(getlines(1,True)[0])

