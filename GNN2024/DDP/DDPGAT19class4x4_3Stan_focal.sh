#!/bin/bash
#PBS -q queue_name          # 使用するキュー名（例: gpu）
#PBS -l select=1:ncpus=4:ngpus=4  # 必要なリソースの選択 (1ノード, 4 CPU, 4 GPU)
#PBS -l walltime=48:00:00   # 実行時間の上限
#PBS -N GAT_training        # ジョブ名
#PBS -o output.log          # 標準出力ログファイル
#PBS -e error.log           # 標準エラーログファイル
#PBS -j oe                  # 標準出力とエラーログを1つのファイルにまとめる
# モジュールの読み込み（必要に応じて）
# module load cuda/11.x      # CUDAバージョン
# module load python/3.x     # Pythonバージョン
# 作業ディレクトリに移動
# cd $PBS_O_WORKDIR
# ジョブ実行中にシェル接続が切れても実行が継続されるようにする
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=12355 DDPGAT19class4x4_3Stan_focal_loss.py --hidden_channels 64 --learning_rate 0.01 --batch_size 64 --weight_decay 0.0005 --patience 200 --epoch 1' > output_$(date +%Y%m%d_%H%M%S).log 2>&1 &
