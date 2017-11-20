from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
import numpy as np
from tensorflow.models.tutorials.rnn.ptb import reader

data_path = './simple-examples/data/'
# raw_data=reader.ptb_raw_data('./simple-examples/data/')
# train_data,valid_data,test_data,_=reader.ptb_raw_data(data_path)
# print(len(train_data))
# print(train_data[:100])

flags = tf.flags  #可以在命令行python *.py --model=  --data_path
logging = tf.logging
                    #第一个是参数名，第二个是默认值，第三个是注释
flags.DEFINE_string('model','small','A type of model,options:small,medium,large')
flags.DEFINE_string('data_path','./simple-examples/data/','data_path')
flags.DEFINE_bool('use_fp16',False,'Train using 16bit floats insetad of 32 bit floats')

FLAGS = flags.FLAGS  # 可以使用FLAGS.model来调用变量 model的值

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

# init_scale = 0.1 #相关初始的参数值为随机均匀分布[-0.1,0.1]
# learning_rate = 0.1 #学习速率，在文本循环次数超过max_epoch以后逐渐降低
# max_grad_norm = 5 #用户控制梯度的膨胀，如果梯度向量的L2模超过5则等比例缩小
# num_layers = 2 #lstm层数
# num_steps = 20 #单个数据中，序列的长度
# hidden_size=200 #隐藏层中单元数目
# max_epoch=4 #epoch<max_epoch lr_decay=1 epoch>max_epoch时lr_decay逐渐减小
# max_max_epoch=13 #整个文本的循环次数
# keep_prob=1.0 #用于dropout.每批数据输入时神经网络中的每个单元会以1-keep_prob的概率不工作，可以防止过拟合
# lr_decay=0.5 #学习速率衰减
# batch_size=20 #每批数据规模
# vocab_size=10000 #词典规模，总共有10k个词

'''
PTBModel 
1.多层LSTM结构的构建
2.输入预处理
3.LSTM循环
4.损失函数计算
5.梯度函数计算和修剪
'''
class PTBModel(object):
    def __init__(self,is_training,config):
        '''
        is_training: 是否要进行训练.如果is_training=False,则不会进行参数的修正
        '''
        self.batch_size=batch_size = config.batch_size
        self.num_steps = num_steps=config.num_steps

        size = config.hidden_size
        vocab_size=config.vocab_size

        self._input_data=tf.placeholder(tf.int32,[batch_size,num_steps])
        self._targets=tf.placeholder(tf.int32,[batch_size,num_steps])

        #定义两个占位符，输入和输出都是[batch_size,num_steps]
        #多层LSTM结构和状态初始化

        lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(size,forget_bias=0.0,state_is_tuple=True)
        if is_training and config.keep_prob<1:
            lstm_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=config.keep_prob)
        cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*config.num_layers,state_is_tuple=True)

        self._initial_state=cell.zero_state(batch_size,data_type())
        
        #输入预处理
        with tf.device('/cpu:0'):
            embedding=tf.get_variable('embedding',[vocab_size,size])    #embedding 是随机初始化在训练中不断改变的
            inputs=tf.nn.embedding_lookup(embedding,self._input_data)   #返回的tensor是batch_size*num_steps*size
           #对input data dropout
        if is_training and config.keep_prob<1:
            inputs=tf.nn.dropout(inputs,config.keep_prob)

        
        #output
        outputs=[]
        state=self._initial_state #state 表示各个batch中的状态
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step>0: tf.get_variable_scope().reuse_variables()  #每次循环都更新参数
                # cell_out: [batch, hidden_size]
                (cell_output,state)=cell(inputs[:,time_step,:],state) #这里的inputs是所有batch,第time_step个单词，所有维度
                outputs.append(cell_output) # output: shape[num_steps][batch,hidden_size]
        
        output=tf.reshape(tf.concat(1,outputs),[-1,size]) #转化为[batch,hidden_size*num_steps]，然后reshape, 成[batch*num_steps, hidden_size]
        ## softmax_w , shape=[hidden_size, vocab_size], 用于将distributed表示的单词转化为one-hot表示
        softmax_w=tf.get_variable('softmax_w',[size,vocab_size],dtype=data_type())
        softmax_b=tf.get_variable('sofmax_b',[vocab_size],dtype=data_type())
        ## [batch*num_steps, vocab_size] 从隐藏语义转化成完全表示
        #上面代码的上半部分主要用来将多层lstm单元的输出转化成one-hot表示的向量
        logits=tf.matmul(output,softmax_w)+softmax_b #加上权重之后的最后输出

        #loss,shape=[batch*steps]
        #带权重的交叉熵就算
        loss=tf.contrib.seq2seq.sequence_loss(
            [logits], # output [batch*num_steps, vocab_size]
            [tf.reshape(self._targets,[-1])], # target, [batch_size, num_steps] 然后展开成一维【列表】,n行一列
            [tf.ones([batch_size*num_steps])] # 计算得到平均每批batch的误差
        )
        self._cost=cost=tf.reduce_sum(loss)/batch_size
        self._final_state=state

        if not is_training: return

        self._lr=tf.Variable(0.0,trainable=False) #定义学习速率，并将其设为不可训练
        #如果想在训练过程中调节learning rate的话，生成一个lr的variable，但是trainable=False，也就是不进行求导。
        tvars=tf.trainable_variables() #获得全部可训练参数
        # clip_by_global_norm: 梯度衰减，具体算法为t_list[i] * clip_norm / max(global_norm, clip_norm)
        # 这里gradients求导，ys和xs都是张量
        # 返回一个长为len(xs)的张量，其中的每个元素都是\grad{\frac{dy}{dx}}
        # clip_by_global_norm 用于控制梯度膨胀,前两个参数t_list, global_norm, 则
        # t_list[i] * clip_norm / max(global_norm, clip_norm)
        # 其中 global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
        grads,_=tf.clip_by_global_norm(tf.gradients(cost,tvars),config.max_grad_norm)

         # 梯度下降优化，指定学习速率
        optimizer=tf.train.GradientDescentOptimizer(self._lr)
        self._train_op=optimizer.apply_gradients(zip(grads,tvars)) #将梯度应用于变量
        self._new_lr=tf.placeholder(tf.float32,shape=[],name='new_learning_rate')
        self._lr_update=tf.assign(self._lr,self._new_lr) #定义新的节点 ,使用_new_lr来更新_lr

    def assign_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self._new_lr:lr_value}) #定义一个函数来运行新的节点
    
    @property #属性函数(property)
    def input_data(self):
        return self._input_data
    @property
    def targets(self):
        return self._targets
    @property
    def cost(self):
        return self._cost
    @property
    def initial_state(self):
        return self._initial_state
    @property
    def final_state(self):
        return self._final_state
    @property
    def lr(self):
        return self._lr
    @property
    def train_op(self):
        return self._train_op
class SmallConfig(object):
    """Small config."""
    init_scale = 0.1        #
    learning_rate = 1.0     # 学习速率
    max_grad_norm = 5       # 用于控制梯度膨胀，
    num_layers = 2          # lstm层数
    num_steps = 20          # 单个数据中，序列的长度。
    hidden_size = 200       # 隐藏层规模
    max_epoch = 4           # epoch<max_epoch时，lr_decay值=1,epoch>max_epoch时,lr_decay逐渐减小
    max_max_epoch = 13      # 指的是整个文本循环13遍。
    keep_prob = 1.0
    lr_decay = 0.5          # 学习速率衰减
    batch_size = 20         # 每批数据的规模，每批有20个。
    vocab_size = 10000      # 词典规模，总共10K个词

class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000

class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000

class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

def run_epoch(session,model,data,eval_op,verbose=False):
    epoch_size=(len(data)//model.batch_size-1)//model.num_steps
    start_time=time.time()
    costs=0.0
    iters=0
    state=session.run(model.initial_state)
    fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
          feed_dict[c] = state[i].c
          feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.num_steps

        if verbose and step % (epoch_size // 10) == 10:
          print("%.3f perplexity: %.3f speed: %.0f wps" %
                (step * 1.0 / epoch_size, np.exp(costs / iters),
                 iters * model.batch_size / (time.time() - start_time)))

    # for step,(x,y) in enumerate(reader.ptb_iterator(data,model.batch_size,model.num_steps)):
    #     fetches=[model.cost,model.final_state,eval_op]
    #     feed_dict={}
    #     feed_dict[model.input_data]=x
    #     feed_dict[model.targets]=y
    #     for i,(c,h) in enumerate(model.initial_state):
    #         feed_dict[c]=state[i].c
    #         feed_dict[h]=state[i].h
    #     cost,state,_=session.run(fetches,feed_dict)
    #     costs+=cost
    #     iters+=model.num_steps
    #     if verbose and step %(epoch_size//10)==10: #每个epoch要输出10个perplexity值
    #         print('%.3f perplexity: %.3f speed: %.0f wps'%
    #         (step*1.0/epoch_size,np.exp(costs/iters),
    #         iters*model.batch_size/(time.time()-start_time)))

    return np.exp(costs/iters)

def get_config():
    if FLAGS.model=='small':
        return SmallConfig
    elif FLAGS.model=='medium':
        return MediumConfig
    elif FLAGS.model=='large':
        return LargeConfig
    elif FLAGS.mosel=='test':
        return TestConfig
    else:
        raise ValueError('Invalid model: %s',FLAGS.model)

if __name__=='__main__':
    if not FLAGS.data_path:
        raise ValueError('Must set --data_path to PTB data directory')
    print(FLAGS.data_path)

    raw_data=reader.ptb_raw_data(FLAGS.data_path)
    train_data,valid_data,test_data,_=raw_data

    config=get_config()
    eval_config=get_config()
    eval_config.batch_size=1
    eval_config.num_steps=1

    with tf.Graph().as_default(),tf.Session() as session:
        initializer=tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        with tf.variable_scope('model',reuse=None,initializer=initializer):
            m=PTBModel(is_training=True,config=config) #训练模型
        with tf.variable_scope('model',reuse=True,initializer=initializer):
            mvalid=PTBModel(is_training=False,config=config) #交叉验证模型和测试模型
            mtest=PTBModel(is_training=False,config=eval_config)
        
        #sumnary_writer=tf.train.SummaryWriter('/tmp/lstm_logs',session.graph)
        sumnary_writer=tf.train.summary.FileWriter('/tmp/lstm_logs',session.graph)
        tf.initialize_all_variables.run()

        for i in range(config.max_max_epoch):
            lr_decay=config.lr_decay**max(i-config.max_epoch,0.0)
            m.assign_lr(session,config.learning_rate*lr_decay)
            print('Epoch: %d Learning rate: %.3f'%(i+1,session.run(m.lr)))
            train_perplexity=run_epoch(session,m,train_data,m.train_op,verbose=True)
            print('Epoch:%d Train Perplexity: %.3f'%(i+1,train_perplexity))
            valid_perplexity=run_epoch(session,mvalid,valid_data,tf.no_op())
            print('Epoch:%d Valid Perplexity:%.3f' %(i+1,valid_data))
        test_perplexity=run_epoch(session,mtest,test_data,tf.no_op())
        print('Test Perplexity: %.3f' %test_perplexity)
