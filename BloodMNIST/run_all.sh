#!/usr/bin/env bash
# Executa toda a Atividade 3 em sequencia. Logs em results/logs/.
set -e
DATA=${DATA:-data/bloodmnist.npz}
W=${WEIGHTS:-weights/resnet18_a1_in1k.pth}
mkdir -p results/logs
python3 a_mlp.py --data $DATA            2>&1 | tee results/logs/a.log
python3 b_cnn_sweep.py --data $DATA      2>&1 | tee results/logs/b.log
python3 c_cnn_best.py --data $DATA       2>&1 | tee results/logs/c.log
python3 d_transfer.py --data $DATA --weights $W 2>&1 | tee results/logs/d.log
python3 e_compare.py                     2>&1 | tee results/logs/e.log
