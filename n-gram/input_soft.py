import numpy as np
import pandas as pd

from Bio import SeqIO

from tqdm import tqdm


df = ['Membrane', 'Nucleus', 'Cytoplasm', 'Cell membrane', 'Endoplasmic reticulum', 'Golgi apparatus',
'Mitochondrion', 'Secreted', 'Chromosome', 'Plastid', 'Peroxisome', 'Lysosome', 'cytoskeleton', 'Endosome']


def data_go_sl(seq_data, go_data):
	dic = {} 
	go = list(open(go_data, "r"))

	print (str(go_data)+':')

	for line in go[1:]:
		if 'SUBCELLULAR LOCATION' in line:
			#tmpp = sl_d(line)
			tmp = line.strip('\n').split('\t') 
			if tmp[0] not in dic.keys(): 
				dic[tmp[0]] = [] 
			dic[tmp[0]].append(tmp[-2].split('; '))
			#dic[tmp[0]].append(tmpp)
			label = []
			for i in df:
				if i in line:
					label.append(1)
				else:
					label.append(0)
			dic[tmp[0]].append(label)

	print ('go & label finish')

	seqen = list(SeqIO.parse(seq_data, "fasta"))
	for i in range(len(seqen)):
		tmm = seqen[i].id.split('|')[1] 
		if tmm in dic.keys():  
			dic[tmm].append(seqen[i].seq) 
	print ('seqen finish') 

	return dic


def select(data, fun, n):
	di_both = {} 
	for i in data.keys():  
		di_both[i] = []  
		go = data[i][0] 
		for j in fun:  
			if j in go:  
				di_both[i].append(j)
	print ('go enter finish')

	dic = {}  
	for i in di_both.keys():  
		if len(di_both[i]) > n:  
			dic[i] = data[i]

	print ('select finish')

	return dic

def fea_go_sl(x, fun):
    go_dic = []
    sl_dic = []
    for i in x.keys(): 
        go = {} 
        for j in range(len(fun)): 
            if fun[j] in x[i][0]: 
                go[fun[j]] = 1 
            else: 
                go[fun[j]] = 0 
        go_dic.append(go)

        sl_dic.append(x[i][1])

    go_dic = pd.DataFrame(go_dic, columns=fun)
    go_dic = np.array(go_dic) 

    sl_dic = np.array(sl_dic)

    return go_dic, sl_dic  
 

