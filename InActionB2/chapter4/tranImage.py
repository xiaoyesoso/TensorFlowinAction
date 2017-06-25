from PIL import Image
import numpy as np
import tensorflow as tf
import scipy.io as sio

STYLE_WEIGHT = 1
CONTENT_WEIGHT = 1
STYLE_LAYERS = ['relu1_2','relu2_2','relu3_2']
CONTENT_LAYERS = ['relu1_2']
_vgg_params = None

def vgg_params():
    global _vgg_params
    if _vgg_params is None:
        _vgg_params = sio.loadmat('imagenet-vgg-verydeep-19.mat')
    return _vgg_params

def vgg19(input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4','pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )

    weights = vgg_params()['layers'][0]
    net = input_image
    network = {}
    for i,name in enumerate(layers):
        layer_type = name[:4]
        if layer_type == 'conv':
            kernels,bias = weights[i][0][0][0][0]
            kernels = np.transpose(kernels,(1,0,2,3))
            conv = tf.nn.conv2d(net,tf.constant(kernels),strides=(1,1,1,1),padding='SAME',name=name)
            net = tf.nn.bias_add(conv,bias.reshape(-1))
            net = tf.nn.relu(net)
        elif layer_type == 'pool':
            net = tf.nn.max_pool(net,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
        network[name] = net

    return network

def content_loss(target_features,content_features):
    _,height,width,channel = map(lambda i:i.value,content_features.get_shape())
    print ('content_features.get_shape() : ')
    print (content_features.get_shape())
    content_size = height * width * channel
    return tf.nn.l2_loss(target_features - content_features) / content_size

def style_loss(target_features,style_features):
    _,height,width,channel = map(lambda i:i.value,target_features.get_shape())
    print ('target_features.get_shape() : ')
    print (target_features.get_shape())
    size = height * width * channel
    target_features = tf.reshape(target_features,(-1,channel))
    target_gram = tf.matmul(tf.transpose(target_features),target_features) / size

    style_features = tf.reshape(style_features,(-1,channel))
    style_gram = tf.matmul(tf.transpose(style_features),style_features) / size

    return tf.nn.l2_loss(target_gram - style_gram) / size

def loss_function(style_image,content_image,target_image):
    style_features = vgg19([style_image])
    content_features = vgg19([content_image])
    target_features = vgg19([target_image])
    loss = 0.0
    for layer in CONTENT_LAYERS:
        loss += CONTENT_WEIGHT * content_loss(target_features[layer],content_features[layer])

    for layer in STYLE_LAYERS:
        loss += STYLE_WEIGHT * style_loss(target_features[layer],style_features[layer])

    return loss



def stylize(style_image,content_image,learning_rate=0.1,epochs=500):
    target = tf.Variable(tf.random_normal(content_image.shape),dtype=tf.float32)
    style_input = tf.constant(style_image,dtype=tf.float32)
    content_input = tf.constant(content_image, dtype=tf.float32)
    cost = loss_function(style_input,content_input,target)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        tf.initialize_all_variables().run()
        for i in range(epochs):
            _,loss,target_image = sess.run([train_op,cost,target])
            print("iter:%d,loss:%.9f" % (i, loss))
            if (i+1) % 100 == 0:
                image = np.clip(target_image + 128,0,255).astype(np.uint8)
                Image.fromarray(image).save("./neural_me_%d.jpg" % (i + 1))


if __name__ == '__main__':
    style = Image.open('star.jpg')
    style = np.array(style).astype(np.float32) - 128.0
    content = Image.open('me.jpg')
    content = np.array(content).astype(np.float32) - 128.0
    stylize(style,content,0.5,500)
    #print(content.shape)
    #print(style.shape)
