#!/bin/bash
for i in {1..1201..50}
do
   echo "processing:" $i
   ./cmake-build-debug/run_DE_INT /home/jingwei/dataset/hamlyn/6/images_rectified $i output 5 1 12 12 0.05 0.95 0 10 0.60 0 1 0 0 10 10 5 0 3 1.6 2
done
