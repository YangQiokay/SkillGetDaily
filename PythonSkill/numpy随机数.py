#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 7/2/17 8:42 AM

@Author: 礼貌的笑了笑，寸步不让
"""
import numpy as np
sizes = [3,4]
print [np.random.randn(y, 1) for y in sizes[:]]

# np.random.randn函数生成均值为0， 标准差为1的高斯分布