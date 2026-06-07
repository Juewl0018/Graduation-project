import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import argparse

# 导入你原代码中的结构
from fedavg_design_mnist_eval import SimpleMLP, build_train_dataset, build_test_dataset, FLConfig


def train_shadow_model(model, dataloader, device, epochs=5, lr=0.01):
    """训练影子模型：不需要DP，不需要HE，就是普通的本地训练"""
    print("开始训练影子模型...")
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Shadow Model Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")
    return model


def get_model_outputs(model, dataloader, device):
    """提取模型的输出特征：Loss 和 Maximum Confidence"""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    all_losses = []
    all_confs = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            confs, _ = torch.max(probs, dim=1)

            all_losses.extend(loss.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    return np.array(all_losses), np.array(all_confs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model-path", type=str, required=True,
                        help="路径：联邦学习最终生成的 final_global_model.pt")
    args = parser.parse_args()

    device = torch.device("cpu")
    cfg = FLConfig(dataset="mnist")

    # ================= 1. 数据严格切分 =================
    print("正在切分数据集...")
    full_train_set = build_train_dataset(cfg, client_id=0)
    full_test_set = build_test_dataset(cfg)

    # 影子模型的数据集 (Shadow Train 标签为 IN, Shadow Test 标签为 OUT)
    shadow_train_loader = DataLoader(Subset(full_train_set, range(20000, 40000)), batch_size=64, shuffle=True)
    shadow_test_loader = DataLoader(Subset(full_test_set, range(5000, 10000)), batch_size=64, shuffle=False)

    # 目标模型的数据集 (Target Train 标签为 IN, Target Test 标签为 OUT)
    target_train_loader = DataLoader(Subset(full_train_set, range(0, 20000)), batch_size=64, shuffle=False)
    target_test_loader = DataLoader(Subset(full_test_set, range(0, 5000)), batch_size=64, shuffle=False)

    # ================= 2. 训练影子模型 =================
    shadow_model = SimpleMLP(cfg.input_dim, cfg.num_classes).to(device)
    shadow_model = train_shadow_model(shadow_model, shadow_train_loader, device, epochs=5)

    # ================= 3. 构建攻击模型的数据集 =================
    print("\n提取影子模型特征，构建攻击数据集...")
    # 影子模型的 IN 数据特征
    s_in_loss, s_in_conf = get_model_outputs(shadow_model, shadow_train_loader, device)
    # 影子模型的 OUT 数据特征
    s_out_loss, s_out_conf = get_model_outputs(shadow_model, shadow_test_loader, device)

    X_shadow = np.vstack((
        np.column_stack((s_in_loss, s_in_conf)),
        np.column_stack((s_out_loss, s_out_conf))
    ))
    y_shadow = np.concatenate((np.ones(len(s_in_loss)), np.zeros(len(s_out_loss))))

    print("训练攻击模型 (Logistic Regression)...")
    attack_model = LogisticRegression(class_weight='balanced')
    attack_model.fit(X_shadow, y_shadow)

    # ================= 4. 对目标模型实施攻击 =================
    print("\n加载目标模型（你的联邦学习模型）...")
    target_model = SimpleMLP(cfg.input_dim, cfg.num_classes).to(device)
    target_model.load_state_dict(torch.load(args.target_model_path))

    print("提取目标模型特征...")
    t_in_loss, t_in_conf = get_model_outputs(target_model, target_train_loader, device)
    t_out_loss, t_out_conf = get_model_outputs(target_model, target_test_loader, device)

    X_target = np.vstack((
        np.column_stack((t_in_loss, t_in_conf)),
        np.column_stack((t_out_loss, t_out_conf))
    ))
    y_target_true = np.concatenate((np.ones(len(t_in_loss)), np.zeros(len(t_out_loss))))

    print("\n========== MIA 最终攻击结果 ==========")
    y_target_pred = attack_model.predict(X_target)
    mia_acc = accuracy_score(y_target_true, y_target_pred)
    print(f"针对目标模型的攻击准确率: {mia_acc * 100:.2f}%")
    print("======================================")


if __name__ == "__main__":
    main()