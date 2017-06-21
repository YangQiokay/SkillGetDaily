#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 9:52 PM

@Author: 礼貌的笑了笑，寸步不让
"""


# 题目：利用递归方法求5!。
# 程序分析：递归公式：fn=fn_1*4!

def fact(n):
    sum = 0
    if n == 0:
        sum = 1
    else:
        sum = n * fact(n - 1)
    return sum


for i in range(6):
    print "%d! = %d" % (i, fact(i))
