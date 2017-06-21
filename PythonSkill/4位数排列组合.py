#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 8:00 AM

@Author: 礼貌的笑了笑，寸步不让
"""

# 题目：有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？
# 程序分析：可填在百位、十位、个位的数字都是1、2、3、4。组成所有的排列后再去 掉不满足条件的排列。

num = []
for a in range(1, 5):
    for b in range(1, 5):
        for c in range(1, 5):
            if (a != b) and (a != c) and (b != c):
                # num.append([a, b, c])
                num.append(a * 100 + b * 10 + c)

print "总数量", len(num)
print num
