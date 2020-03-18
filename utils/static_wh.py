import os

labels_path = "/home/dongpeijie/datasets/dimtargetSingle/labels/data_sum"

_min = 100.
_max = -100.

for label in os.listdir(labels_path):
    label_full_path = os.path.join(labels_path, label)
    with open(label_full_path, 'r') as f:
        content = f.readlines()
        for line in content:
            print(line)
            _, _, _, w, h = line.split()
            w = float(w)
            h = float(h)
            _min = min(_min, w)
            _min = min(_min, h)
            _max = max(_max, w)
            _max = max(_max, h)

print("min:%.3f  \nmax:%.3f" % (_min*416,_max*416))