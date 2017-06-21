#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 9:54 AM

@Author: 礼貌的笑了笑，寸步不让
"""
#
# l = []
# for i in range(3):
#     x = int(raw_input('integer:\n'))
#     l.append(x)
#
# l.sort()
# print l

# 方法二：
x = int(raw_input("x:"))
y = int(raw_input("y:"))
z = int(raw_input("z:"))

a = {"x": x, "y": y, "z": z}
print "*" * 20
for w in sorted(a, key=a.get):
    # get返回指定键的值，也就是按照值排序
    print w, a[w]
