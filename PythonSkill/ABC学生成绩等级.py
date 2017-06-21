#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 6/21/17 3:46 PM

@Author: 礼貌的笑了笑，寸步不让
"""

score = int(raw_input("请输入学生成绩：\n"))
if score >= 90:
    grade = 'A'
elif score >= 60:
    grade = 'B'
else:
    grade = 'C'
print "%d属于%s" % (score, grade)
