# conding=utf-8
# 导入需要的包
import os, getConfig, jieba

gConfig = {}
gConfig = getConfig.get_config()

# 配置源文本的路径

conv_path = gConfig['resource_data']
# 判断文件是否存在
if not os.path.exists(conv_path):
    exit()
# 识别读取训练集的数据并存入一个列表中
# a 打开文件
# b 读取文件中的内容， 并对文件的数据进行初步处理
# c 找到想要的数据保存起来
convs = [] # 存储对话的列表
with open(conv_path, encoding='utf-8') as f:
    one_conv = []
    for line in f:
        line = line.strip('\n').replace('/','') # 去除换行符, 并将原文中的已经分词的标记去掉，重新用jieba分词
        if line == '':
            continue
        if line[0] == gConfig['e']:
            if one_conv:
                convs.append(one_conv)
            one_conv = []
        elif line[0] == gConfig['m']:
            one_conv.append(line.split(' ')[1]) # 保存一次完整的对话

# 接下来骂我们需要对训练集中的对话进行分类， 分为文具和大局，或者叫上文、下文、主要作为Encoder和Decoder的训练集，一般分为几个步骤：

# 1 按照语句的顺序分为文具和大局，根据行数的就行来判断

# 2 在存储语句的时候对语句使用jieba分词

# 把对话分为问句和答句两个部分
seq = []
for conv in convs:
    if len(conv) == 1:
        continue
    # 因为默认是一问一答，所以需要对数据进行粗裁剪，对话行数要为偶数
    if len(conv) % 2 == 0:
        conv = conv[-1]
    for i in range(len(conv)):
        if i % 2 == 0:
            # 使用jieba分词
            conv[i] == " ".join(jieba.cut(conv[i]))
            conv[i+1] == " ".join(jieba.cut(conv[i+1]))
            # 因为i 是从0开始，因此偶数行为问句，奇数行为答句
            seq.append(conv[i] + '\t' +conv[i+1])
# 新建一个文件用于存储处理好的数据，作为训练集
seq_train = open(gConfig['seq_data'],'w')

# 将处理好的数据保存到文件中

for i in range(len(seq)):
    seq_train.write(seq[i] + '\n')
    if i % 1000 == 0:
        print(len(range(len(seq))), '处理进度', i)

# 保存并退出
seq_train.close()
