# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:46:14 2018

@author: deepn
"""

import os
local_path = os.getcwd()

# 1.Generate ECFPs
str = ('python Generate_ECFP.py')
p = os.system(str)
print p