import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from kg_config import TRANSE_DIR, TRANSE_VECTORS_PATH


class TransE(nn.Module):
    def __init__(self, num_ent, num_rel, dim=128, margin=1.0):
        super().__init__()
        self.ent_emb = nn.Embedding(num_ent, dim)
        self.rel_emb = nn.Embedding(num_rel, dim)
        self.margin = margin
        self.criterion = nn.MarginRankingLoss(margin=margin)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def _calc(self, h, r, t):
        return torch.norm(h + r - t, p=1, dim=1)

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_score = self._calc(self.ent_emb(pos_h), self.rel_emb(pos_r), self.ent_emb(pos_t))
        neg_score = self._calc(self.ent_emb(neg_h), self.rel_emb(neg_r), self.ent_emb(neg_t))
        # MarginRankingLoss: max(0, -y * (x1 - x2) + margin) -> 我们希望 neg > pos + margin
        return self.criterion(neg_score, pos_score, torch.tensor([1.0]).to(pos_h.device))


def main():
    print(">>> [Step 2] 开始 TransE 嵌入训练...")

    # 加载数据
    with open(os.path.join(TRANSE_DIR, 'entity2id.json')) as f:
        ent2id = json.load(f)
    with open(os.path.join(TRANSE_DIR, 'relation2id.json')) as f:
        rel2id = json.load(f)
    triples = np.load(os.path.join(TRANSE_DIR, 'train_triples.npy'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransE(len(ent2id), len(rel2id)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 训练循环
    triples_tensor = torch.tensor(triples, dtype=torch.long).to(device)
    batch_size = 1024
    epochs = 100

    print(f"    实体: {len(ent2id)}, 关系: {len(rel2id)}, 样本: {len(triples)}")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(triples_tensor))
        total_loss = 0

        for i in range(0, len(triples_tensor), batch_size):
            batch = triples_tensor[perm[i:i + batch_size]]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # 简单的负采样：随机替换尾实体
            neg_t = torch.randint(0, len(ent2id), (len(batch),)).to(device)

            loss = model(h, r, t, h, r, neg_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

    # 导出向量 (供 Step 4 推理使用)
    print("    导出向量...")
    output = {
        "ent_embeddings": model.ent_emb.weight.detach().cpu().numpy().tolist(),
        "rel_embeddings": model.rel_emb.weight.detach().cpu().numpy().tolist(),
        "entity2id": ent2id,
        "relation2id": rel2id
    }
    with open(TRANSE_VECTORS_PATH, 'w') as f:
        json.dump(output, f)
    print(f">>> [Step 2] 向量已保存至: {TRANSE_VECTORS_PATH}")


if __name__ == "__main__":
    main()