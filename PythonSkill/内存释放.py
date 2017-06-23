#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/22/17 9:57 AM

@Author: 礼貌的笑了笑，寸步不让
"""

# python话说会自己管理内存，实际上，对于占用很大内存的对象，并不会马上释放。举例，a=range(10000*10000)，会发现内存飙升一个多G，del a 或者a=[]都不能将内存降下来。。

# del 可以删除多个变量，del a,b,c,d
# 办法：

import gc
a = range(10000000)
del a
gc.collect()#马上内存就释放了。
