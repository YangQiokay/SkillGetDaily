#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 7/2/17 9:39 AM

@Author: 礼貌的笑了笑，寸步不让
"""

import mnist_loader
import network

trainingdata, validationdata, testdata = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(trainingdata, 30, 10, 3.0, testdata)
