import os
import numpy as np
import tensorflow as tf
import argparse
import os
import hdfs3
import pandas as pd

from urllib.parse import urlparse
from os.path import basename, join, exists

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="", help='input path')
parser.add_argument('--output', type=str, default="", help='output path')
parser.add_argument('--model', type=str, default="", help='model path')

parser.add_argument('--train', type=int, default=200, help='Number of times to train')
parser.add_argument('--modelname', type=str, default="model", help='model name')

FLAGS, unparsed = parser.parse_known_args()

mPATH = FLAGS.output

# 데이터 입력
path_list = FLAGS.input.split(os.path.sep)
print(path_list[0], path_list[2], path_list[4])
master, port = path_list[2].split(':')
print(master, port)
hdfs = hdfs3.HDFileSystem(master, port=int(port), user= path_list[4])
input_file_path = '/' + os.path.join(*path_list[3:])
with hdfs.open(input_file_path) as f:
    data = pd.read_csv(f, nrows=None, header=None)

num_rows = data.shape[0]


# 데이터 가공
x_data = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
y_data = np.array([0, 1])
data_dump = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0, 1])
for i in range(num_rows - 10):
    x_data_tmp = np.array([data.iloc[i][1], 
                    data.iloc[i+1][1],
                    data.iloc[i+2][1], 
                    data.iloc[i+3][1], 
                    data.iloc[i+4][1], 
                    data.iloc[i+5][1], 
                    data.iloc[i+6][1], 
                    data.iloc[i+7][1]])
    y_data_tmp = np.array([data.iloc[i+3][0],
                    1 - data.iloc[i+3][0]])
    data_dump_tmp = np.array([data.iloc[i][1], 
                    data.iloc[i+1][1],
                    data.iloc[i+2][1], 
                    data.iloc[i+3][1], 
                    data.iloc[i+4][1], 
                    data.iloc[i+5][1], 
                    data.iloc[i+6][1], 
                    data.iloc[i+7][1],
                    data.iloc[i+3][0],
                    1 - data.iloc[i+3][0]])
    x_data = np.vstack((x_data, x_data_tmp))
    y_data = np.vstack((y_data, y_data_tmp))
    data_dump = np.vstack((data_dump, data_dump_tmp))
#np.savetxt("/home/csle/x_data.csv", x_data, delimiter=",")
#np.savetxt("/home/csle/y_data.csv", y_data, delimiter=",")
np.savetxt("/home/csle/ksb-csle/dump_data.csv", data_dump, delimiter=",")


#########
# 신경망 모델 구성
######
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, 8], name = 'x_data')

with tf.name_scope('output'):
    Y = tf.placeholder(tf.float32, [None, 2], name = 'y_data')


global_step = tf.Variable(0, trainable=False, name='global_step')

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([8, 100], -1., 1.), name = 'W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([100, 50], -1., 1.), name = 'W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_uniform([50, 10], -1., 1.), name = 'W3')
    L3 = tf.nn.relu(tf.matmul(L2, W3))

with tf.name_scope('layer4'):
    W4 = tf.Variable(tf.random_uniform([10, 2], -1., 1.), name = 'W4')
    model = tf.matmul(L3, W4)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)



saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if tf.gfile.Exists(mPATH):
        tf.gfile.DeleteRecursively(mPATH)
    tf.gfile.MakeDirs(mPATH)
    writer = tf.summary.FileWriter(mPATH, sess.graph)
    

    for step in range(FLAGS.train):
        sess.run(train_op, feed_dict={X: x_data, Y: y_data})

        print('Step: %d, ' % sess.run(global_step),
              'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    
    save_path = saver.save(sess, mPATH + '/' +  FLAGS.modelname + '.ckpt')

    ################################
    ### tensorflow serving model ###
    ################################
    import shutil
    def _to_proto(tensors):
        protos = {}
        for k, v in tensors.items():
            protos[k] = tf.saved_model.utils.build_tensor_info(v)
        return protos


    #tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
    #tensor_info_y = tf.saved_model.utils.build_tensor_info(Y1)

    parsed_url = urlparse(mPATH + '/' +  FLAGS.modelname)
    base_path = parsed_url.path

    model_version = '1'
    model_path = join(base_path, model_version)
    if exists(model_path):
        shutil.rmtree(model_path)
    builder = tf.saved_model.builder.SavedModelBuilder(model_path)
    
    target = tf.argmax(model, 1)
    signature_name1 = "predict"
    input_tensors = {"X": X}
    output_tensors = {"Y": target}
    signature1 = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=_to_proto(input_tensors),
        outputs=_to_proto(output_tensors),
        #inputs=input_tensors,
        #outputs=output_tensors,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
    
    builder.add_meta_graph_and_variables(
        sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            signature_name1: signature1
        },
        assets_collection=None,
        legacy_init_op=None,
        clear_devices=None,
        main_op=None)
    builder.save(as_text=False)
    print("모델 파일은 %s 에 저장되었습니다." % (mPATH + '/' +  FLAGS.modelname))
    x_data_test =  np.array([[0.28,0.28,0.28,0.28,0.28,0.28,0.28,0.28],
                        [0.33,0.33,0.31,0.33,0.33,0.5,0.7,0.9],
                        [0.8,0,0.9,0.9,0,0,0,0],
                        [0,0,0,0,0.28,0.28,0.28,0.28]])
    print("학습 테스트 #1")
    xp = tf.argmax(model, 1)
    print(sess.run(xp, feed_dict={X: x_data_test}))
