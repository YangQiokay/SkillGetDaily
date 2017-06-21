#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 3:07 PM

@Author: 礼貌的笑了笑，寸步不让
"""


# 题目：古典问题：有一对兔子，从出生后第3个月起每个月都生一对兔子，小兔子长到第三个月后每个月又生一对兔子，假如兔子都不死，问每个月的兔子总数为多少？
# 程序分析：兔子的规律为数列1,1,2,3,5,8,13,21....

# f1 = 1
# f2 = 1
# for i in range(1, 22):
#     print "%12ld %12ld" % (f1, f2),
#     if (i % 3) == 0:
#         print " "
#     f1 = f1 + f2
#     f2 = f1 + f2


# 方法二：递归

def fib(n):
    if n == 1 or n == 2:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


print fib(21)
