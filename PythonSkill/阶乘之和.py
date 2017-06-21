#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 9:44 PM

@Author: 礼貌的笑了笑，寸步不让
"""

# 题目：求1+2!+3!+...+20!的和。
# 程序分析：此程序只是把累加变成了累乘。

s = 0
t = 1
for n in range(1, 21):
    t *= n
    s += t
print "1! + 2! + 3! + ... + 20! = %d" %s