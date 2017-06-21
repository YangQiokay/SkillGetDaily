#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 9:56 PM

@Author: 礼貌的笑了笑，寸步不让
"""


def output(s, l):
    if l == 0:
        return
    print s[l - 1],
    output(s, l - 1)


s = raw_input("Input a String")
l = len(s)
output(s, l)
