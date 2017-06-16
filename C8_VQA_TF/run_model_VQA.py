#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os, h5py
import time
import json

    
#####################################################
#                 Global Parameters            #  
#####################################################
print("Loading parameters ...")

MODE = 'Train'

# Data input setting
input_img_h5 = 'data/data_img.h5'
input_ques_h5 = 'data/data_prepro.h5'
input_json = 'data/data_prepro.json'

# Train Parameters setting
learning_rate = 0.0003            # learning rate for rmsprop
#starter_learning_rate = 3e-4
learning_rate_decay_start = -1        # at what iteration to start decaying learning rate? (-1 = dont)
batch_size = 500            # batch_size for each iterations
input_embedding_size = 200        # he encoding size of each token in the vocabulary
rnn_size = 512                # size of the rnn in number of hidden nodes in each layer
rnn_layer = 2                # number of the rnn layer
dim_image = 4096
dim_hidden = 1024 #1024            # size of the common embedding vector
num_output = 1000            # number of output answers
img_norm = 1                # normalize the image feature. 1 = normalize, 0 = not normalize
decay_factor = 0.99997592083

# Check point
checkpoint_path = 'model_save/'

# misc
gpu_id = 0
max_itr = 150000
n_epochs = 300
max_words_q = 26
num_answer = 1000
#####################################################

class Answer_Generator():
    def __init__(self, rnn_size, rnn_layer, batch_size, input_embedding_size, dim_image, dim_hidden, max_words_q, vocabulary_size, drop_out_rate):

        # 网络参数 network parameters
        self.rnn_size = rnn_size
        self.rnn_layer = rnn_layer
        self.batch_size = batch_size
        self.input_embedding_size = input_embedding_size
        self.dim_image = dim_image
        self.dim_hidden = dim_hidden
        self.max_words_q = max_words_q
        self.vocabulary_size = vocabulary_size    
        self.drop_out_rate = drop_out_rate
        self.question_state_dim = 2*rnn_size*rnn_layer
        
        # 随机初始器random uniform initializer.
        self.initializer = tf.random_uniform_initializer(minval=-0.08,maxval=0.08)


    def build_model(self, mode = 'Train'):
        # 训练输入placeholders
        image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
        question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])
        label = tf.placeholder(tf.int64, [self.batch_size,]) 
        
        # 文本特征embedding
        with tf.variable_scope("seq_embedding"):
            # question-embedding
            self.embed_ques_W = tf.get_variable(shape=[self.vocabulary_size, self.input_embedding_size], 
                                                name='embed_ques_W',
                                                initializer=self.initializer)
        # 2层LSTM的问题question序列编码器
        # encoder: 2-layer LSTM 
        lstms = []
        for _ in range(rnn_layer):
            lstm = tf.contrib.rnn.LSTMCell(num_units = rnn_size)
            lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = 1 - self.drop_out_rate)
            lstms.append(lstm_dropout)
        self.stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstms)        
                
        # 对问题question进行编码操作
        with tf.variable_scope("lstm_encoder", initializer=self.initializer):
            # get zero state
            state = self.stacked_lstm.zero_state(batch_size = self.batch_size, dtype=tf.float32)
            loss = 0.0
            for i in range(max_words_q):
                if i==0:
                    ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
                else:
                    tf.get_variable_scope().reuse_variables()
                    ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question[:,i-1])
                # 
                ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self.drop_out_rate)
                ques_emb = tf.tanh(ques_emb_drop)
                # exexute stacked lstm
                output, state = self.stacked_lstm(ques_emb, state)
        

        # 对图片特征、问题特征进行embed、融合、推断
        with tf.variable_scope("feature_embed"):
            # 问题特征 embedding 操作
            self.embed_state_W = tf.get_variable(shape=[self.question_state_dim, self.dim_hidden],
                                         name='embed_state_W',
                                         initializer=self.initializer)
            self.embed_state_b = tf.get_variable(shape=[self.dim_hidden], 
                                         name='embed_state_b',
                                         initializer=self.initializer)
            # reshape state from [2,layer,batch,dim] to [batch, 2*layer*dim]
            state = tf.transpose(state,perm=[2,0,1,3])
            state = tf.reshape(state, [-1, self.question_state_dim])
            # multimodal (fusing question & image)
            state_drop = tf.nn.dropout(state, 1-self.drop_out_rate)
            state_linear = tf.nn.xw_plus_b(state_drop, self.embed_state_W, self.embed_state_b)
            state_emb = tf.tanh(state_linear)
            
            # 图片特征 embedding操作
            self.embed_image_W = tf.get_variable(shape=[self.dim_image, self.dim_hidden], 
                                             name='embed_image_W',
                                             initializer=self.initializer)
            self.embed_image_b = tf.get_variable(shape=[self.dim_hidden], 
                                             name='embed_image_b',
                                             initializer=self.initializer)
            image_drop = tf.nn.dropout(image, 1-self.drop_out_rate)
            image_linear = tf.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b)
            image_emb = tf.tanh(image_linear)
            
            # 融合推断操作score-embedding
            self.embed_scor_W = tf.get_variable(shape=[self.dim_hidden, num_output], 
                                            name='embed_scor_W',
                                            initializer=self.initializer)
            self.embed_scor_b = tf.get_variable(shape=[num_output], 
                                            name='embed_scor_b',
                                            initializer=self.initializer)
            scores = tf.multiply(state_emb, image_emb)
            scores_drop = tf.nn.dropout(scores, 1-self.drop_out_rate)
            scores_emb = tf.nn.xw_plus_b(scores_drop, self.embed_scor_W, self.embed_scor_b)         
    
        if mode == 'Train':
            # 交叉熵loss函数Calculate cross entropy
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_emb, labels=label)
            # 合并loss
            loss = tf.reduce_mean(cross_entropy)
            # return
            return loss, image, question, label
        else:
            return scores_emb, image, question            


def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
    return v

def get_data():

    dataset = {}
    train_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # 加载图片特征(VGG19倒数第二层4096维特征) load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_train')
        img_feature = np.array(tem)
        
    # 加载标注特征 load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_train')
        # convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
        # answer is 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

    # 数据对齐
    print('question aligning')
    train_data['question'] = right_align(train_data['question'], train_data['length_q'])

    # 图片特征L2归一化
    print('Normalizing image feature')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1))))

    return dataset, img_feature, train_data

def train():
    # 加载数据集
    print('loading dataset...')
    dataset, img_feature, train_data = get_data()
    num_train = train_data['question'].shape[0]
    vocabulary_size = len(dataset['ix_to_word'].keys())
    print('vocabulary_size : ' + str(vocabulary_size))

    # 构建模型
    print('constructing  model...')
    model = Answer_Generator(
            rnn_size = rnn_size,
            rnn_layer = rnn_layer,
            batch_size = batch_size,
            input_embedding_size = input_embedding_size,
            dim_image = dim_image,
            dim_hidden = dim_hidden,
            max_words_q = max_words_q,    
            vocabulary_size = vocabulary_size,
            drop_out_rate = 0.5)
    tf_loss, tf_image, tf_question, tf_label = model.build_model()

    # 获取session和模型存储器
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(max_to_keep=100)

    # Adam优化器
    lr = tf.Variable(learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    
    # 剪切梯度的计算操做gradient clipping
    tvars = tf.trainable_variables()
    grads = tf.gradients(tf_loss,tvars)
    clipped_grads,_ = tf.clip_by_global_norm(grads,5.0)
    train_op = opt.apply_gradients(zip(clipped_grads,tvars))

    # 执行初始化execute initialization
    tf.global_variables_initializer().run()

    #开始训练
    tStart_total = time.time()
    print('start training...')
    for itr in range(max_itr):
        tStart = time.time()
        
        # 随机洗牌，获取batch训练数据
        # shuffle the training data
        index = np.random.random_integers(0, num_train-1, batch_size)
        current_question = train_data['question'][index,:]
        current_answers = train_data['answers'][index]
        current_img_list = train_data['img_list'][index]
        current_img = img_feature[current_img_list,:]

        # 执行训练
        # do the training process
        _, loss = sess.run(
                [train_op, tf_loss],
                feed_dict={
                    tf_image: current_img,
                    tf_question: current_question,
                    tf_label: current_answers
                    })
    
        # 更新学习率
        current_learning_rate = lr*decay_factor
        lr.assign(current_learning_rate).eval()
       
        tStop = time.time()
        
        # 定期打印log和存储模型
        if np.mod(itr, 100) == 0:
           print("Iteration: ", itr, " Loss: ", loss, " Learning Rate: ", lr.eval())
           print("Time Cost:", round(tStop - tStart,2), "s")
        if np.mod(itr, 15000) == 0:
           print("Iteration ", itr, " is done. Saving the model ...")
           saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=itr)

    # 训练结束，存储模型
    print("Finally, saving the model ...")
    saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    print("Total Time Cost:", round(tStop_total - tStart_total, 2), "s")



def get_data_test():
    dataset = {}
    test_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # 加载图片特征(VGG19倒数第二层4096维特征) load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        tem = hf.get('images_test')
        img_feature = np.array(tem)
        
    # 加载标注特征 load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_test')
        # convert into 0~82459
        test_data['img_list'] = np.array(tem)-1
        # quiestion id
        tem = hf.get('question_id_test')
        test_data['ques_id'] = np.array(tem)
        # MC_answer_test
        tem = hf.get('MC_ans_test')
        test_data['MC_ans_test'] = np.array(tem)

    # 数据对齐
    print('question aligning')
    test_data['question'] = right_align(test_data['question'], test_data['length_q'])

    # L2归一化图片特征
    print('Normalizing image feature')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1))))

    return dataset, img_feature, test_data

def test(model_path):
    # 加载数据集
    print('loading dataset...')
    dataset, img_feature, test_data = get_data_test()
    num_test = test_data['question'].shape[0]
    vocabulary_size = len(dataset['ix_to_word'].keys())
    print('vocabulary_size : ' + str(vocabulary_size))

    # 构建模型
    model = Answer_Generator(
            rnn_size = rnn_size,
            rnn_layer = rnn_layer,
            batch_size = batch_size,
            input_embedding_size = input_embedding_size,
            dim_image = dim_image,
            dim_hidden = dim_hidden,
            max_words_q = max_words_q,
            vocabulary_size = vocabulary_size,
            drop_out_rate = 0)
    tf_answer, tf_image, tf_question, = model.build_model('Test')


    # 建立session、模型存储器、初始化模型
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # 开始测试
    tStart_total = time.time()
    result = []
    for current_batch_start_idx in range(0,num_test-1,batch_size):
        tStart = time.time()
        # set data into current*
        if current_batch_start_idx + batch_size < num_test:
            current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+batch_size)
        else:
            current_batch_file_idx = range(current_batch_start_idx,num_test)

        current_question = test_data['question'][current_batch_file_idx,:]
        current_length_q = test_data['length_q'][current_batch_file_idx]
        current_img_list = test_data['img_list'][current_batch_file_idx]
        current_ques_id  = test_data['ques_id'][current_batch_file_idx]
        current_img = img_feature[current_img_list,:] # (batch_size, dim_image)

        # deal with the last batch
        if(len(current_img)<500):
                pad_img = np.zeros((500-len(current_img),dim_image),dtype=np.int)
                pad_q = np.zeros((500-len(current_img),max_words_q),dtype=np.int)
                pad_q_len = np.zeros(500-len(current_length_q),dtype=np.int)
                pad_q_id = np.zeros(500-len(current_length_q),dtype=np.int)
                pad_ques_id = np.zeros(500-len(current_length_q),dtype=np.int)
                pad_img_list = np.zeros(500-len(current_length_q),dtype=np.int)
                current_img = np.concatenate((current_img, pad_img))
                current_question = np.concatenate((current_question, pad_q))
                current_length_q = np.concatenate((current_length_q, pad_q_len))
                current_ques_id = np.concatenate((current_ques_id, pad_q_id))
                current_img_list = np.concatenate((current_img_list, pad_img_list))

        # 执行测试
        generated_ans = sess.run(
                tf_answer,
                feed_dict={
                    tf_image: current_img,
                    tf_question: current_question
                    })
        # 获取结果
        top_ans = np.argmax(generated_ans, axis=1)

        # 收集结果
        # initialize json list
        for i in range(0,500):
            ans = dataset['ix_to_ans'][str(top_ans[i]+1)]
            if(current_ques_id[i] == 0):
                continue
            result.append({u'answer': ans, u'question_id': str(current_ques_id[i])})

        tStop = time.time()
        print ("Testing batch: ", current_batch_file_idx[0])
        print ("Time Cost:", round(tStop - tStart,2), "s")
        
    #测试结束
    print ("Testing done.")
    tStop_total = time.time()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")
    # 存储结果
    # Save to JSON
    print('Saving result...')
    my_list = list(result)
    json.dump(my_list,open('data/data.json','w'))

if __name__ == '__main__':
    if MODE == 'Train':
        with tf.device('/gpu:'+str(0)):
            train()
    else:
        with tf.device('/gpu:'+str(1)):
            test(model_path=('model_save/model-', max_itr))
    
