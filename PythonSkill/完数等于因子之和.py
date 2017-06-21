#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 4:04 PM

@Author: 礼貌的笑了笑，寸步不让
"""

# 一个数如果恰好等于它的因子之和，这个数就称为"完数"。例如6=1＋2＋3.编程找出1000以内的所有完数。


for i in range(1, 5):
    sum = 0
    for j in range(1, i):
        if i % j == 0:
            sum += j
            print i,j
    if sum == i:
        print (i)
