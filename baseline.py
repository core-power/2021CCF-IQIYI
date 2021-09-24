import json
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from tqdm import tqdm
from keras.utils import multi_gpu_model
import os
from tensorflow.python.ops import array_ops
from keras import backend as K
import tensorflow as tf

def seed_everything(seed=0):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

SEED = 1234
seed_everything(SEED)
is_train = True
gpus = '0,1,2'
num_classes = 24
maxlen = 256
n_gpu = len(gpus.split(','))
batch_size = 64
epochs = 20
learn_rating=2e-5
model_name = 'roberta_base'
model_save = model_name+'.weights'
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
if model_name == 'nezha_base':
    # bert配置
    config_path = '../nezha_base/bert_config.json'
    checkpoint_path = '../nezha_base/model.ckpt'
    dict_path = '../nezha_base/vocab.txt'
if model_name == 'roberta_large':
    # bert配置
    config_path = '../roberta_wwm_large/bert_config.json'
    checkpoint_path = '../roberta_wwm_large/bert_model.ckpt'
    dict_path = '../roberta_wwm_large/vocab.txt'
if model_name == 'roberta_base':
    # bert配置
     config_path = '../roberta_wwm_base/bert_config.json'
     checkpoint_path = '../roberta_wwm_base/bert_model.ckpt'
     dict_path = '../roberta_wwm_base/vocab.txt'
if model_name == 'nezha_large':
    # bert配置
    config_path = '../nezha_large/bert_config.json'
    checkpoint_path = '../nezha_large/model.ckpt'
    dict_path = '../nezha_large/vocab.txt'
if model_name == 'nezha_wwm_large':
    # bert配置
    config_path = '../nezha_wwm_large/bert_config.json'
    checkpoint_path = '../nezha_wwm_large/model.ckpt'
    dict_path = '../nezha_wwm_large/vocab.txt'
if model_name=='roberta_zh_large':
    # bert配置
    config_path = '../roberta_zh_large/bert_config_large.json'
    checkpoint_path = '../roberta_zh_large/roberta_zh_large_model.ckpt'
    dict_path ='../roberta_zh_large/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            text, labels = l.split('\t')[0],[int(i) for i in l.split('\t')[1].split(',')]
            t0, t1, t2, t3, t4, t5 = [0] * 4, [0] * 4, [0] * 4, [0] * 4, [0] * 4, [0] * 4
            t0[labels[0]] = 1
            t1[labels[1]] = 1
            t2[labels[2]] = 1
            t3[labels[3]] = 1
            t4[labels[4]] = 1
            t5[labels[5]] = 1
            label = t0 + t1 + t2 + t3 + t4 + t5
            D.append((text, label))
    return D
     

# 加载数据集
train_data = load_data(
    '/home/maxin/ccf/train.tsv'
)
valid_data = load_data(
    '/home/maxin/ccf/test.tsv'
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
with tf.device('/cpu:0'):
    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
        model=model_name.split('_')[0],
    )

    output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = Dense(
        units=num_classes,
        activation='sigmoid',
        kernel_initializer=bert.initializer
    )(output)

    model = keras.models.Model(bert.model.input, output)
    model.summary()
if is_train==True:
    m_model = multi_gpu_model(model, gpus=n_gpu)

def multi_category_focal_loss2(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha控制真值y_true为1/0时的权重
        1的权重为alpha, 0的权重为1-alpha
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
        尝试将alpha调大,鼓励模型进行预测出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss2_fixed
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed
if is_train==True:
    m_model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learn_rating),
        metrics=['accuracy'],
    )

def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数

if is_train==True:
    # 写好函数后，启用对抗训练只需要一行代码
    adversarial_training(m_model, 'Embedding-Token', 1)


def evaluate(data):
    total, right = 0., 0.
    for x_true, targe in tqdm(data):
        token_ids, segment_ids = tokenizer.encode(x_true, maxlen=maxlen)
        pred = m_model.predict([[token_ids], [segment_ids]])[0].tolist()
        y_pred = []
        y_true = []
        for i in range(0, len(pred), 4):
                x1 = np.array(pred[i: i + 4])
                x2 = np.array(targe[i:i+4])
                y_true.append(str(x2.argmax()))
                y_pred.append(str(x1.argmax()))
        total += len(y_true)
        n = 0
        for i in range(len(y_true)):
            if y_true[i]==y_pred[i]:
                n+=1
        right += n
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_data)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            m_model.save_weights(model_save)
            m_model.load_weights(model_save)
            model.save_weights(model_save)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


def model_predict(file,save_path):
    s = open(save_path, 'w', encoding='utf-8')
    s.write('id'+'\t'+'emotion'+'\n')
    with open(file,'r',encoding='utf-8') as f:
        for l in tqdm(f.readlines()[1:]):
            text = '（'+'描述角色是：'+l.split('\t')[2]+'）'+l.split('\t')[1]
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            pred = model.predict([[token_ids], [segment_ids]])[0].tolist()
            y_pred = []
            for i in range(0, len(pred), 4):
                x1 = pred[i: i + 4]
                y_pred.append(str(x1.index(max(x1))))
            s.write(l.split('\t')[0]+'\t'+','.join(y_pred)+'\n')
    f.close()
    s.close()


if __name__ == '__main__':
    if is_train==True:
        evaluator = Evaluator()

        m_model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )
    else:
        model.load_weights(model_save)
        model_predict('test_dataset.tsv','result.tsv')

