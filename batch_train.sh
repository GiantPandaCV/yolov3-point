#!/bin/bash

python train.py --cfg cfg/dt-6a-spp-RF.cfg --batch-size 64

mkdir weights/dt-6a-spp-RF
mv weights/*.pt weights/dt-6a-spp-RF
mv results* weights/dt-6a-spp-RF




python train.py --cfg cfg/dt-6a-spp-11.cfg --batch-size 64

mkdir weights/dt-6a-spp-11
mv weights/*.pt weights/dt-6a-spp-11
mv results* weights/dt-6a-spp-11




python train.py --cfg cfg/dt-6a-spp-15.cfg --batch-size 64

mkdir weights/dt-6a-spp-15
mv weights/*.pt weights/dt-6a-spp-15
mv results* weights/dt-6a-spp-15




python train.py --cfg cfg/dt-6a-spp-11-15.cfg --batch-size 64

mkdir weights/dt-6a-spp-11-15
mv weights/*.pt weights/dt-6a-spp-11-15
mv results* weights/dt-6a-spp-11-15