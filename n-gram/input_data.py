import numpy as np
import pandas as pd

from Bio import SeqIO

from tqdm import tqdm

from data_seq import n_dgram, protein_seq1, fea   

from go_func_level_reader import reader_go, dic_level

from input_soft import data_go_sl, select, fea_go_sl

lunion = ['A','C','E','D','G','F','I','H','K','M','L','N','Q','P','S','R','T','W','V','Y']

UN = n_dgram(lunion)

#go level information download
dic_go, go_all = dic_level(reader_go('./lable_go_level.txt'))

#Choose the samples with go-terms at least 3 to be the training instance.
def data_go_sl_sel(seq, go, go_type, num):
	data = data_go_sl(seq, go)
	sel_type = select(data, go_type, num)

	return sel_type


data_h = data_go_sl_sel('uniprot-(Human).fasta', 'uniprot-(Human).tab', go_all, 3)
data_a = data_go_sl_sel('uniprot-(Arabidopsis).fasta', 'uniprot-(Arabidopsis).tab', go_all, 3) 
data_m = data_go_sl_sel('uniprot-(Mouse).fasta', 'uniprot-(Mouse).tab', go_all, 3)
data_r = data_go_sl_sel('uniprot-(Rat).fasta', 'uniprot-(Rat).tab', go_all, 3)
data_c = data_go_sl_sel('uniprot-(cerevisiae).fasta', 'uniprot-(cerevisiae).tab', go_all, 3)
data_f = data_go_sl_sel('uniprot-(fly).fasta', 'uniprot-(fly).tab', go_all, 3)
data_fu = data_go_sl_sel('uniprot-(fungus).fasta', 'uniprot-(fungus).tab', go_all, 3)
data_ri = data_go_sl_sel('uniprot-(Rice).fasta', 'uniprot-(Rice).tab', go_all, 3)


sel_all = dict(data_h, **data_a)
sel_all = dict(sel_all, **data_m)
sel_all = dict(sel_all, **data_r)
sel_all = dict(sel_all, **data_c) 
sel_all = dict(sel_all, **data_f)
sel_all = dict(sel_all, **data_fu)
sel_all = dict(sel_all, **data_ri)   


def seq_go_sl_all(x, go_all, lunion, UN):

	seqen = protein_seq1(x, lunion)   
	seq = fea(seqen, lunion)/420
	go, sl = fea_go_sl(x, go_all)

	data = np.c_[seq, go, sl]

	return data

data_all = seq_go_sl_all(sel_all, go_all, lunion, UN)


