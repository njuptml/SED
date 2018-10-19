# -*- coding: utf-8 -*-
"""
Created on Tue Oct 02 17:31:39 2018

@author: Benli
"""
import os 
import sys
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np
import xlrd
import scipy.io as scio

gpcr_name = sys.argv[1]
gpcr_length = sys.argv[2]
gpcr_radius = sys.argv[3]

local_path = os.path.dirname(os.getcwd())
gpcr_path = local_path +'\\data\\'+ gpcr_name

gpcr_diameter = int(gpcr_radius) * 2


def smiles_to_fps(data, fp_length, fp_radius):
    return stringlist2intarray(np.array([smile_to_fp(s, fp_length, fp_radius) for s in data]))

def smile_to_fp(s, fp_length, fp_radius):
    m = Chem.MolFromSmiles(s)
    return (AllChem.GetMorganFingerprintAsBitVect(
            m, fp_radius, nBits=fp_length, invariants=[1]*m.GetNumAtoms(), useFeatures=False)).ToBitString()

def stringlist2intarray(A):
    '''This function will convert from a list of strings "10010101" into in integer numpy array.'''
    return np.array([list(s) for s in A], dtype=int)

if __name__ == "__main__":
    
    excel = xlrd.open_workbook(gpcr_path+'\\Input_Smiles.xlsx')
    #获取第一个sheet
    sheet = excel.sheets()[0]
    
    #打印第j列数据
    x1 = sheet.col_values(0)
    
    a1 = smiles_to_fps(x1, int(gpcr_length), int(gpcr_radius))
    
    scio.savemat(gpcr_name + '_ECFP' + str(gpcr_diameter) + '_' + gpcr_length, {"X":a1})  
#    scio.savemat('temp_ECFP12_10240', {"X":a1})  