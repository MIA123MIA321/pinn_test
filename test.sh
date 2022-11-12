#!/bin/bash


MAXITER=10000
N=250000
n=1600
m='5,6'
grids=256
k=2
heatmap_dir='heatmap/'
relative_dir='relative/'
TMP_PATH='.tmp.log'
GPU='yes'
if [ $GPU == 'yes' ]; then
	PYPATH='pinn-gpu.py'
	heatmap_dir='heatmap/gpu_'$MAXITER'_'
	relative_dir='relative/gpu_'$MAXITER'_'
else
	PYPATH='pinn.py'
	heatmap_dir='heatmap/cpu_'$MAXITER'_'
	relative_dir='relative/cpu_'$MAXITER'_'
fi


echo > ${TMP_PATH} &&
nohup python -u \
${PYPATH} \
--maxiter ${MAXITER} \
--N ${N} \
--n ${n} \
--m ${m} \
--grids ${grids} \
--k ${k} \
--heatmap_dir ${heatmap_dir} \
--relative_dir ${relative_dir} \
>> ${TMP_PATH} 2>&1 &


# tail -f /data/liuziyang/pde_solver/pinn/.tmp.log
# 来查看运算进度