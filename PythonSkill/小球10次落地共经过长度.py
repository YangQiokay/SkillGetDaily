#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 4:17 PM

@Author: 礼貌的笑了笑，寸步不让
"""
# 题目：一球从100米高度自由落下，每次落地后反跳回原高度的一半；再落下，求它在第10次落地时，共经过多少米？第10次反弹多高？

from __future__ import division

tour = []
height_times = []

height = 100
time = 10

for i in range(1, time + 1):
    if i == 1:
        tour.append(height)
    else:
        tour.append(2 * height)
    height /= 2
    height_times.append(height)
print "Total: tour = {}".format(sum(tour))
print "第10次反弹高度：height_10 = {}".format(height_times[-1])
