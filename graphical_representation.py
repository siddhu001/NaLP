import pickle
relation_name_dict={}
entity_name_dict={}
relation_doub_dict={}
entity_doub_dict={}
for line in open("data/JF17K_version1/train.txt"):
	line1=(line.rstrip().split("\t"))
	print(line1)
	if (line1[0] not in relation_name_dict):
		relation_name_dict[line1[0]]=len(relation_name_dict)
	if (relation_name_dict[line1[0]] not in relation_doub_dict):
		relation_doub_dict[relation_name_dict[line1[0]]]={}
	for j in range(1,len(line1)):	
		if (line1[j] not in entity_name_dict):
			entity_name_dict[line1[j]]=len(entity_name_dict)
		if (entity_name_dict[line1[j]] not in relation_doub_dict[relation_name_dict[line1[0]]]):
			relation_doub_dict[relation_name_dict[line1[0]]][entity_name_dict[line1[j]]]=1
		if (entity_name_dict[line1[j]] not in entity_doub_dict):
			entity_doub_dict[entity_name_dict[line1[j]]]={}
		if (relation_name_dict[line1[0]] not in entity_doub_dict[entity_name_dict[line1[j]]]):
			entity_doub_dict[entity_name_dict[line1[j]]][relation_name_dict[line1[0]]]=1
	# break
entity_dict={}
relation_dict={}
for k in entity_doub_dict:
	entity_dict[k]=list(entity_doub_dict[k].keys())
for k in relation_doub_dict:
	relation_dict[k]=list(relation_doub_dict[k].keys())
pickle.dump(relation_name_dict,open("relation_name_dict.pkl","wb"))
pickle.dump(entity_name_dict,open("entity_name_dict.pkl","wb"))
file_write=open("p_a_list_train.txt","w")
for k in relation_dict:
	file_write.write(str(k)+":")
	for j in range(len(relation_dict[k])):
		if (j==(len(relation_dict[k])-1)):
			file_write.write(str(relation_dict[k][j])+"\n")
		else:
			file_write.write(str(relation_dict[k][j])+",")
	file_write.write("\n")
file_write=open("a_p_list_train.txt","w")
for k in entity_dict:
	file_write.write(str(k)+":")
	for j in range(len(entity_dict[k])):
		if (j==(len(entity_dict[k])-1)):
			file_write.write(str(entity_dict[k][j])+"\n")
		else:
			file_write.write(str(entity_dict[k][j])+",")
	# file_write.write("\n")
# print(entity_dict)