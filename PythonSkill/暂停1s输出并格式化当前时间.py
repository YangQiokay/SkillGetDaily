#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 1:58 PM

@Author: 礼貌的笑了笑，寸步不让
"""

import time

print time.strftime('%Y-%m-%d %H:%M:%s', time.localtime(time.time()))

time.sleep(1)

print time.strftime('%Y-%m-%d %H:%M:%s', time.localtime(time.time()))
