import numpy as np
import pickle
import time
import sys

train_file=open("data/JF17K_version1/train.txt")
train_final_file=open("JF17K_version1_TransE_train.txt","w")
for line in train_file:
	line1=line.rstrip().split("\t")
	for k in range(len(line1[1:])):
		graph_line=line1[0]+"\t"+line1[0]+str(k)+"\t"+line1[k+1]
		train_final_file.write(graph_line+"\n")

train_file=open("data/JF17K_version1/test.txt")
train_final_file=open("JF17K_version1_TransE_test.txt","w")
for line in train_file:
	line2=line.rstrip().split("\t")
	line1=line2[1:]
	for k in range(len(line1[1:])):
		graph_line=line1[0]+"\t"+line1[0]+str(k)+"\t"+line1[k+1]
		train_final_file.write(graph_line+"\n")