import numpy as np

def reader_go(x):

	num = []
	num2 = []
	with open(x, 'r') as f:
		for line in f.readlines():
			num.append(line.split('\t')[1:])

	for i in num[0]:
		num2.append(i.strip('\n'))

	return num2


def split_wave(string):
    if '~' in string:
        string=string.split('~')

    return string



def dic_level(x):
	
	d = {}
	for line in x:
		tmps = split_wave(line)		
		if tmps[1]=='Level':
			lev = 0
		else:
			lev = int(tmps[1].strip('Level'))

		if lev not in d.keys():
			d[lev] = []
		d[lev].append(tmps[0])

	go = []
	for key in sorted(d.keys()):
		print (str(key) +": "+ str(len(d[key])))
		for j in d[key]:
			go.append(j)

	return d, go

def go_fu(x):
	
	cc = []
	mf = []	
	bp = []	
	for i in x:
		tmps = split_wave(i)
		if tmps[2]=='C':
			cc.append([tmps[0], tmps[1]])
		elif tmps[2]=='F':
			mf.append([tmps[0], tmps[1]])
		else:
			bp.append([tmps[0], tmps[1]])

	return cc, mf, bp



def dic_fun_level(x):
	d = {}
	for i in range(len(x)):
		if x[i][-1]=='Level':
			lev = 0
		else:
			lev = int(x[i][-1].strip('Level'))

		if lev not in d.keys():
			d[lev] = []
		d[lev].append(x[i][0])

	go = []
	for key in sorted(d.keys()):
		print (str(key) +": "+ str(len(d[key])))
		for j in d[key]:
			go.append(j)

	return d, go


