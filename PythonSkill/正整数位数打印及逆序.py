#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 10:02 PM

@Author: 礼貌的笑了笑，寸步不让
"""

# 题目：给一个不多于5位的正整数，要求：一、求它是几位数，二、逆序打印出各位数字。

num = list(raw_input("请输入最多5位数字"))
print len(num)
num.reverse()
for i in range(len(num)):
    print num[i],