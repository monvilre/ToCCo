#! /usr/bin/env python3
# coding: utf-8
""" This script is to run ToCCo on multiple core for a set of Parameters
writing the last output line in a file.
The output need to be of the form: [parameter,value] """
__author__ = "Rémy Monville"

import sys
import numpy as np
import os
import ast
from subprocess import Popen, PIPE

filename = "./dissipation.dat"
params = np.logspace(2, 8, 80) # list of parameters
print('script launched')

cmds_list = [['python', 'ToCCo.py', str(par)] for par in params] # create command lines
procs_list = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmds_list]

Val_Diss = np.zeros((0, 2)) # Empty list
# Val_P =Val_Diss
for proc in procs_list:
    proc.wait()
    stdout_Diss = (proc.stdout.readlines())[-1].decode("utf-8")
    # stdout_P = (proc.stdout.readlines())[-2].decode("utf-8")
    stdout_Diss = ast.literal_eval(stdout_Diss)
    # stdout_P = ast.literal_eval(stdout_P)
    out_Diss = np.array(stdout_Diss)
    # out_P = np.array(stdout_P)
    Val_Diss = np.vstack((Val_Diss, out_Diss))
    # Val_P = np.vstack((Val_P,out_P))
files = open(filename, "w+") # writing file
files.write(str(Val_Diss))
files.close()
