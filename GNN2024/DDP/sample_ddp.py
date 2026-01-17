import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# シンプルなモデル定義
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

def setup():
    # 環境変数から必要な情報を取得
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '12355')

    # プロセスグループの初期化
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

    # デバイス設定
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    return device

def cleanup():
    dist.destroy_process_group()

def main():
    device = setup()

    # モデルの初期化とDDPのラッピング
    model = SimpleModel().to(device)
    ddp_model = DDP(model, device_ids=[device.index] if torch.cuda.is_available() else None)

    # 損失関数とオプティマイザの定義
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # ダミーデータ
    inputs = torch.randn(20, 10).to(device)
    targets = torch.randn(20, 10).to(device)

    # トレーニングループ
    for epoch in range(5):
        ddp_model.train()
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if dist.get_rank() == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    cleanup()
    if dist.get_rank() == 0:
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
