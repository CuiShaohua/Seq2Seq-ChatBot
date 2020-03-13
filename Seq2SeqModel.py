import tensorflow as tf
import getConfig
tf.compat.v1.enable_eager_execution()

gConfig = {}
gConfig = getConfig.get_config(config_file='seq2seq.ini')
# Encoder model
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size # 批大小
        self.enc_units = enc_units # 神经元数量
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        #
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        # GRU

    def call(self, x, hiddden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hiddden)
        return output, state

    # 初始化隐藏状态
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))

# 定义Attention机制
class BahdanauAttention(tf.keras.Model):

    # 定义初始化函数
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # 初始化定义权重网络层W1, W2 以及最后的打分网络V，最终打分结果作为注意力的权重值
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    # 定义调用的函数，输入输出的逻辑变缓在这个函数中完成
    def call(self, query, values):
        # hidden shape = (batch_size, hidden size)
        # hidden_with_time_axis shape = (batch size, 1, hidden_size)
        # 计算attiong score
        hidden_with_time_axis = tf.expand_dims(query,1)
        # score的维度shi (batch size, maxlength, hidden_size)
        # 构建评价计算网络结构，受限计算W1和W2，然后将W1和W2的和 经过tanh进行非线性变缓，最后输入打分网络层
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        # attention_weights shape == (batch_size. maxlength, 1)
        # 计算attention_weights的值，我们使用softmax将score的值进行归一化，得到的是总和唯一的各个score值的概率分布
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector 文本向量的维度是(batch_size, hidden_size)
        # 将attention_weights的值与输入文本进行相乘，得到加权过的文本向量
        context_vector = attention_weights * values
        # 将上一步得到的文本向量按行求和，得到最终的文本向量
        context_vector = tf.reduce_sum(context_vector,axis=1)
        #  返回最终的文本向量和注意力权重
        return context_vector, attention_weights

# Decoder

class Decoder(tf.keras.Model):
    #
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        # 初始化批训练数据的大小
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # 初始化定义RNN结构，采用RNN的变种GRU结构
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        # 初始化定义全连接输出层
        self.fc = tf.keras.layers.Dense(vocab_size)
        # 使用Attention机制
        self.attention = BahdanauAttention(self.dec_units)

    # 定义调用函数，输入、输出的逻辑变换在这个函数中完成
    def call(self, x, hidden, enc_output):
        # 解码器输出的维度是（batch_size, maxlength, hidden_size）
        # 根据输入hidden和输出值使用Attention机制计算文本向量和注意力权重，hidden就是编码器输出的编码向量
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # 将x的维度在Embedding之后是(batch sizem ,1, embedding_dim)
        # 对解码器的输入进行Embedding处理
        x = self.embedding(x)
        # 将Embedding之后的向量和 经过Attention后的编码器输出的编码向量进行连接，然后作为输入向量输入到gru中
        x = tf.concat([tf.expand_dims(context_vector,1), x], axis=-1)
        # 将连接之后的编码向量输入gru中得到输出值和state
        output, state = self.gru(x)
        # 将输出的向量进行维度变换，变换成（batch_size, vocab）
        output = tf.reshape(output, (-1, output.shape[2]))
        # 将变换后的向量输入全连接网络中，得到最后的输出值
        outputs = self.fc(output)
        return outputs, state, attention_weights

# 对训练数据的字典大小进行初始化赋值
vocab_inp_size = gConfig['enc_vocab_size']
vocab_tar_size = gConfig['dec_vocab_size']
#对embedding的维度尽心初始化赋值
embedding_dim = gConfig['embedding_dim']
# 对网络层的神经元数量进行初始化赋值
units = gConfig['layer_size']

# 对批训练数据的大小进行初始化赋值
BATCH_SIZE = gConfig['batch_size']

# 实例化Encoder模型
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
# 实例化Attention模型
attention_layer = BahdanauAttention(10)
#实例化Decoder模型
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
# 定义优化器
optimizer = tf.keras.optimizers.Adam()

# 定义整个模型的损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 定义损失函数
def loss_function(real, pred):
    # dropout
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # 计算损失向量
    loss_ = loss_object(real, pred)
    # 转换成mask向量的类型
    mask = tf.cast(mask, dtype=loss_.dtype)
    # 使用mask向量对损失向量进行处理，去除Padding引入的噪声
    loss_ *= mask
    # 返回平均损失函数
    # 将损失函数转化为numpy
    return tf.reduce_mean(loss_)


# 实例化checkoint的方法类， 使用其中的save方法保存训练模型
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

# 定义训练方法， 对输入的数据进行一次循环训练
def train_step(inp, targ, targ_lang, enc_hidden):
    loss = 0
    ## 使用tf,GraientTape记录梯度求导信息
    with tf.GradientTape() as tape:
        # 使用编码器对输入的语句进行编码，得到编码器的编码向量输出enc_output和中间层的输出enc_hidden
        enc_output, enc_hidden= encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        # 构建编码器输入向量，首词会用start对应的字典编码值作为向量的第一个数值，维度是BATCH_SIZE的大小，也就是一次批量训练的语句数量
        dec_input = tf.expand_dims([targ_lang.word_index['start']] * BATCH_SIZE, 1)
        # 开始训练编码器
        for t in range(1, targ.shape[1]):
            # 将构建的编码器输入向量和编码器输出对话中的上一句编码向量作为输入，输入到编码器中，训练解码器
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            # 计算损失值
            loss += loss_function(targ[:,t], predictions)
            # 将对话中的下一句逐步分时作为编码器的输入，这相当于进行移位输入， 先从start标识开始， 逐步输入到对话中的下一句
            dec_input = tf.expand_dims(targ[:,t], 1)

        # 计算批处理的平均损失值
        batch_loss = (loss / int(targ.shape[1]))
        # 计算参数变量
        variables = encoder.trainable_variables + decoder.trainable_variables
        # 计算梯度
        gradients = tape.gradient(loss, variables)
        # 使用优化器优化参数变量的值，以达到拟合的效果
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

