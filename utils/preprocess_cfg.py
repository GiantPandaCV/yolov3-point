import os
import shutil

cfg_path = "./cfg/yolov3-dwconv-cbam.cfg"
save_path = "./cfg/preprocess_cfg/"

new_save_name = os.path.join(save_path,os.path.basename(cfg_path))


f = open(cfg_path, 'r')
lines = f.readlines()

# 去除以#开头的，属于注释部分的内容
# lines = [x for x in lines if x and not x.startswith('#')]
# lines = [x.rstrip().lstrip() for x in lines]

lines_nums = []
layers_nums = []

layer_cnt = -1

for num, line in enumerate(lines):
    if line.startswith('['):
        layer_cnt += 1
        layers_nums.append(layer_cnt)
        lines_nums.append(num+layer_cnt)
        print(line)
        # s = s.join("")
    # s = s.join(line)
for i,num in enumerate(layers_nums):
    print(lines_nums[i], num)
    lines.insert(lines_nums[i]-1, '# layer-%d\n' % (num-1))


fo = open(new_save_name, 'w')
fo.write(''.join(lines))


fo.close()
f.close()
