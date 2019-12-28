#!/usr/bin/env python
# -*- coding: gb18030 -*-
 
"""
FileName: my_set/gen_pretrain_data.py
Date: 2019-12-28 22:51:55
"""
import sys

for line in sys.stdin:
    line = line.strip("\n").split("\t")
    if line[-1] == "1":
        print line[0]
        print line[1]
        print ""
