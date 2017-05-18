#!/usr/bin/env python
# -*- coding=utf8 -*-

import sys
import json
#viterbi算法

class HMM(object):
    """HMM"""
    def __init__(self):
        self.pi = None
        self.trans_matirx = None
        self.obs_matirx = None
        self.hidden_state = None
        self.obs_state = None

    def initialize(self):
        """
        初始化五元组
        """
        #晴、雨
        #干，潮，湿
        self.pi = [0.7, 0.3]
        self.hidden_state = ['SUN', 'RAIN']
        self.obs_state = ['dry', 'damp', 'wet']
        self.obs_matirx = [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6]
        ]
        self.trans_matirx = [
            [0.8, 0.2],
            [0.4, 0.6]
        ]

    def viterbi(self):
        """
        预测算法
        """
        obs = ['dry', 'dry', 'damp', 'wet']
        alpha = [[] for i in range(len(obs))]
        beta = [[] for i in range(len(obs))]
        for i in range(len(obs)):
            for j in range(len(self.hidden_state)):
                alpha[i].append(0)
                beta[i].append(-1)

        idx = self.obs_state.index(obs[0])
        for i, s in enumerate(self.pi):
            prob = s * self.obs_matirx[i][idx]
            alpha[0][i] = prob

        oi = 1 #表示时刻 t
        while oi < len(obs):
            #hi表示该时刻某个隐藏状态
            for hi in range(len(self.hidden_state)):
                #be是历史时刻
                max_p = 0
                for bi, bp in enumerate(alpha[oi-1]):
                    tmp = bp * self.trans_matirx[bi][hi]
                    if tmp > max_p:
                        max_p = tmp
                        beta[oi][hi] = bi
                print max_p
                prob = max_p * self.obs_matirx[hi][self.obs_state.index(obs[oi])]
                alpha[oi][hi] = prob
            oi += 1

        print "alpha:", alpha
        print "beta:", beta
        max_last = 0
        max_i = -1
        for li, tmp in enumerate(alpha[-1]):
            if tmp > max_last:
                max_i = li

        print "last obs state max hidden state:", max_i
        last_res = [max_i]
        for i in range(len(beta)-1, 0, -1):
            before = beta[i][max_i]
            last_res.append(before)
            max_i = before

        print last_res
        last_res.reverse()
        print [self.hidden_state[i] for i in last_res]


def main():
    obj = HMM()
    obj.initialize()
    obj.viterbi()

if __name__ == '__main__':
    main()
