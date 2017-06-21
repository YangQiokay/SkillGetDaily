#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 10:08 PM

@Author: 礼貌的笑了笑，寸步不让
"""

a = raw_input("请输入一串数字")
print type(a)
print a
b = a[::-1]

print b
if a == b:
    print "%s 是回文" % a
else:
    print "%s 不是回文" % a
