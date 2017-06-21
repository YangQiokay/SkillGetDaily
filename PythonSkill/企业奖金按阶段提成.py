#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 8:05 AM

@Author: 礼貌的笑了笑，寸步不让
"""

# 题目：企业发放的奖金根据利润提成。利润(I)低于或等于10万元时，奖金可提10%；利润高于10万元，低于20万元时，低于10万元的部分按10%提成，高于10万元的部分，可提成7.5%；20万到40万之间时，高于20万元的部分，可提成5%；40万到60万之间时高于40万元的部分，可提成3%；60万到100万之间时，高于60万元的部分，可提成1.5%，高于100万元时，超过100万元的部分按1%提成，从键盘输入当月利润I，求应发放奖金总数？
# 程序分析：请利用数轴来分界，定位。注意定义时需把奖金定义成长整型。

# i = int(raw_input("净利润："))
# arr = [100, 60, 40, 20, 10, 0]
# rate = [0.01, 0.015, 0.03, 0.05, 0.075, 0.1]
#
# r = 0
# for idx in range(0, 6):
#     if i > arr[idx]:
#         r += (i - arr[idx]) * rate[idx]
#         print idx, r
#         print (i - arr[idx]) * rate[idx]
#         i = arr[idx]
# print r

# 方法二：

i = int(raw_input("净利润："))
arr = {100: 0.01, 60: 0.015, 40: 0.03, 20: 0.05, 10: 0.075, 0: 0.1}

arr_sorted = sorted(arr.items(), key=lambda d: d[0], reverse=True)
print arr_sorted #此时已经为list了，里面是元组

r = 0
for idx in arr_sorted:
    print idx[0]
    if i > idx[0]:
        r += (i - idx[0]) * idx[1]
        i = idx[0]
print r
