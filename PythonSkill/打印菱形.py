#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 4:39 PM

@Author: 礼貌的笑了笑，寸步不让
"""

# 打印出如下图案（菱形）:
#    *
#   ***
#  *****
# *******
#  *****
#   ***
#    *

# 程序分析：先把图形分成两部分来看待，前四行一个规律，后三行一个规律，利用双重for循环，第一层控制行，第二层控制列。


for i in range(4):
    for j in range(2 - i + 1):
        print  " ",
    for k in range(2 * i + 1):
        print  "*",
    print

for j in range(3):
    for x in range(j+1):
        print " ",
    for y in range(4-2*j+1):
        print "*",
    print
