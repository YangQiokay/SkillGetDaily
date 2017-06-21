#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 3:17 PM

@Author: 礼貌的笑了笑，寸步不让
"""

# 题目：判断101-200之间有多少个素数，并输出所有素数。
# 程序分析：判断素数的方法：用一个数分别去除2到sqrt(这个数)，如果能被整除，则表明此数不是素数，反之是素数。 　　

h = 0
leap = 1
from math import sqrt

for m in range(101, 201):
    k = int(sqrt(m + 1))
    for i in range(2, k + 1):
        if m % i == 0:
            leap = 0
            break
    if leap == 1:
        print m,
        h += 1
    leap = 1

print "\nThe total is %d" % h
