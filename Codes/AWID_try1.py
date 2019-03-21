# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:55:29 2018
@author: mchijtz4:tahmina.zebin@manchester.ac.uk
"""
from tensorflow import set_random_seed
set_random_seed(12)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#Data preparation Stage 
attr=pd.read_csv('H:\\DATASET\\attributes.csv',header=None)
attr1=attr.iloc[:,0].values
AWID_train=pd.read_csv('H:\\DATASET\\AWID-CLS-R-Trn\\1_train.csv',sep=',',encoding= 'utf-8-sig',names =attr1, low_memory=False)


AWID_test=pd.read_csv('H:\\DATASET\\AWID-CLS-R-Tst\\1_test.csv',sep=',',encoding= 'utf-8-sig',names =attr1, low_memory=False)
#import scapy
#from scapy.all import *
#scapy_cap = scapy.all.rdpcap('H:\\DATASET\\AWID-CLS-R-Tst\\1_test.pcap')