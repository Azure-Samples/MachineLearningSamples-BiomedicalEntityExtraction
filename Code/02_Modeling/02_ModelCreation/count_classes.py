# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:47:16 2017

@author: t-akmehr
"""

#train_file = open("LabeledIn//train_labeledin.txt")
#test_file = open("LabeledIn//test.txt")

train_file = open("PBA//train.iob2")
test_file = open("PBA//test.iob2")

#train_file = open("BC5//train_bc5.txt")
#test_file = open("BC5//test_bc5.txt")

#train_file = open("Drugs//train.txt")
#test_file = open("Drugs//test.txt")
classes = []
count = 0 
max_count = 0
for line in train_file:
    line = line.strip()
    if not line:
        max_count = max(max_count, count)
        count = 0
        continue
    count += 1
    word, tag = line.split("\t")
    if tag not in classes:
        classes.append(tag)
        
print(len(classes), max_count)

classes = []
count = 0 
max_count = 0
for line in test_file:
    line = line.strip()
    if not line:
        max_count = max(max_count, count)
        count = 0
        continue
    count += 1
    word, tag = line.split("\t")
    if tag not in classes:
        classes.append(tag)
        
print(len(classes), max_count)

train_file.close()
test_file.close()