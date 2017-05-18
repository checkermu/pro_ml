#!/usr/bin/env python
# -*- coding=utf8 -*-

#https://www.zybuluo.com/hanbingtao/note/433855
import sys

class Perceptron(object):
    """
    感知器类
    """
    def __init__(self, activator, input_num):
        """
        一个激活函数，一个参数(w, b)其实算一个
        """
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def predict(self, input_vec):
        """
        就是用激活函数来计算预测值
        由输入根据定义的激活函数 得到预测的输出
        具体的数值这里就是向量形式  vector
        input_vec = [x1, x2, x3]
        weights = [w1, w2, w3]
        两者相乘；后累加；再输入到激活函数
        """
        return self.activator(
            reduce(lambda a,b : a+b,
            map(lambda (x, w) : x*w,
            zip(input_vec, self.weights)), 0.0) + self.bias)


    def train(self, input_vec, labels, iteration, rate):
        """
        训练更新参数
        """
        for i in range(iteration):
            self.__train_one(input_vec, labels, rate)
            print '权重:', self.weights

    def __train_one(self, input_vec, labels, rate):
        """
        一次迭代所有数据
        """
        #组成样本 （xi, yi)
        samples = zip(input_vec, labels)
        for (input_vec, labels) in samples:
            #梯度迭代公式进行计算更新
            output = self.predict(input_vec)
            self.__update_weights(input_vec, labels, output, rate)

    def __update_weights(self, input_vec, labels, output, rate):
        """
        按照迭代公式更新
        """
        delta = labels - output
        self.weights = map(
            lambda (x, w): w+rate*delta*x,
            zip(input_vec, self.weights))
        #更新bias
        self.bias += rate*delta

    def to_str(self):
            '''
            打印学习到的权重、偏置项
            '''
            return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)


def f(x):
    """
    定义激活函数
    """
    return 1 if x>0 else 0

def get_training_dataset():
    '''
    基于and真值表构建训练数据
    '''
    # 构建训练数据
    # 输入向量列表
    input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels = [1, 0, 0, 0]
    return input_vecs, labels

def train_and_perceptron():
    '''
    使用and真值表训练感知器
    '''
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p = Perceptron(f, 2)
    # 训练，迭代10轮, 学习速率为0.1
    input_vecs, labels = get_training_dataset()
    print len(labels)
    p.train(input_vecs, labels, 10, 0.1)
    #返回训练好的感知器
    return p

if __name__ == '__main__':
    print "ok"
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print and_perception.to_str()
    # 测试
    print '1 and 1 = %d' % and_perception.predict([1, 1])
    print '0 and 0 = %d' % and_perception.predict([0, 0])
    print '1 and 0 = %d' % and_perception.predict([1, 0])
    print '0 and 1 = %d' % and_perception.predict([0, 1])
