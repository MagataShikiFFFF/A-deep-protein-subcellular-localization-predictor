from nltk.util import ngrams
import numpy as np
import pandas as pd
from chainer import Variable
import time

from Bio import SeqIO
import os
import itertools

from tqdm import tqdm


def protein_seq1(data, lunion):  
	pp1 = [] 
	for i in data.keys():  
		line = list(data[i][-1]) 
		for j in list(set(line).difference(lunion)): 
			while j in line: 
				line.remove(j)             
		pp1.append(line) 

	return pp1 


def n_dgram(l):

    lz = []
    for i in l: # 1-gram
    	lz.append(i)
    for i in itertools.product(l,l): # 2-gram
        lz.append(i)
    #for i in itertools.product(l,l,l): # 3-gram
     #   lz.append(i)

    return lz


def freqn(A, lst_test):

	dic = {}

	for l in lst_test:
		dic[l] = 0

	for i, a in enumerate(A):
		str1 = A[i]
		dic[str1] += 1
		try:
			str2 = (A[i],A[i+1])
		#	str3 = (A[i],A[i+1],A[i+2])
			dic[str2] += 1
		#	dic[str3] += 1
		except Exception:
			pass

	return dic

lunion = ['A','C','E','D','G','F','I','H','K','M','L','N','Q','P','S','R','T','W','V','Y']

UN = n_dgram(lunion)

def fea(m, lunion):

	d = []
	for line in m:
		f = freqn(line, lunion)
		d.append(f)

	fea = pd.DataFrame(d, columns=UN)
	fea = np.array(fea)

	return fea

def fea_ex(data, lunion):
    
    pp1 = {}
    for i in data.keys():
        seq = data[i]
        pp1[i] = fea(seq, lunion)

        print (i + 'finished')

    return pp1

