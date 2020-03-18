import os
import shutil
import random

train_txt = "/home/dongpeijie/datasets/dimtargetSingle/2007_train.txt"
test_txt = "/home/dongpeijie/datasets/dimtargetSingle/2007_test.txt"
val_txt = "/home/dongpeijie/datasets/dimtargetSingle/test.txt"

train_out_txt = "/home/dongpeijie/datasets/dimtargetSingle/shuffle_train.txt"
test_out_txt = "/home/dongpeijie/datasets/dimtargetSingle/shuffle_test.txt"

f_train = open(train_txt, "r")
f_test = open(test_txt, "r")
f_val = open(val_txt, "r")

o_train = open(train_out_txt, "w")
o_test = open(test_out_txt, "w")

train_content = f_train.readlines()
test_content = f_test.readlines()
val_content = f_val.readlines()

all_content = [*train_content, *test_content, *val_content]

print(len(train_content), len(test_content), len(all_content))

len_all = len(all_content)

train_percent = 0.8
# train:test = 8:2

train_sample_num = int(len_all * train_percent)
test_sample_num = len_all - train_sample_num

print("Train Sample:%d\nTest Sample:%d\n" % (train_sample_num, test_sample_num))

# print(random.sample(all_content, 10))

sampled_train = random.sample(all_content, train_sample_num)

for i in all_content:
    if i in sampled_train:
        o_train.write(i)
    else:
        o_test.write(i)

print("done")

f_test.close()
f_train.close()
f_val.close()

o_test.close()
o_train.close()