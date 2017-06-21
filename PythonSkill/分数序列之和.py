#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 9:31 PM

@Author: 礼貌的笑了笑，寸步不让
"""

# 题目：有一分数序列：2/1，3/2，5/3，8/5，13/8，21/13...求出这个数列的前20项之和。
# 程序分析：请抓住分子与分母的变化规律。

from __future__ import division

a = 2
b = 1
s = 0
for n in range(1, 21):
    s += a / b
    a, b = a + b, a

print s
