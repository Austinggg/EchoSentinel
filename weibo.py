import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from models.dnf import DNF, DeltaDelayedExponentialDecayScheduler
import os  # 新增导入

class WeiboDataset(Dataset):
    # 修改：接受文件夹路径，遍历所有json文件构建数据集
    def __init__(self, folder_path):
        records = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                # 修改：添加 encoding="utf-8"
                with open(file_path, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                    records.append(data)
        self.features = []
        self.labels = []
        for item in records:
            # 提取P1-P8的值作为特征
            p_values = list(item['general'].values())
            self.features.append(p_values)
            # 假设标签为二进制，如"真实"=1，"虚假"=0
            label = 1 if item.get('label',"") == "1" else 0
            self.labels.append(label)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.labels[idx]])
    
# 添加：包装DNF层的分类器
# 采用正态分布初始化权重
class DNFClassifier(nn.Module):
    def __init__(self, num_preds, num_conjuncts, n_out, delta=0.01, weight_init_type="normal"):
        super(DNFClassifier, self).__init__()
        self.dnf = DNF(num_preds, num_conjuncts, n_out, delta, weight_init_type=weight_init_type)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.dnf(x))

# 参数配置
num_preds = 8       # P1-P8特征维度
num_conjuncts = 20  # 合取项数量（可调整）
n_out = 1           # 二分类输出

model = DNFClassifier(num_preds, num_conjuncts, n_out)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Delta调度器（关键参数需根据实验调整）
delta_scheduler = DeltaDelayedExponentialDecayScheduler(
    initial_delta=0.01,
    delta_decay_delay=100,
    delta_decay_steps=50,
    delta_decay_rate=0.1
)

# ------------------------------------------------------------------
# 数据加载及划分训练集和验证集
dataset = WeiboDataset("./assessment_result")
train_size = int(0.8 * len(dataset))      # 80%作为训练集
val_size = len(dataset) - train_size      # 剩余作为验证集
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# ------------------------------------------------------------------

# 训练参数
num_epochs = 50

for epoch in range(num_epochs):
    # --------------------- 训练阶段 ---------------------
    model.train()
    for step, (inputs, labels) in enumerate(train_loader):
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新delta值
        current_step = epoch * len(train_loader) + step
        delta_scheduler.step(model.dnf, current_step)
    
    # --------------------- 验证阶段 ---------------------
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        corrects = 0  # 新增：累计正确预测数量
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # 计算预测准确性（阈值0.5作为二分类依据）
            predicted = (outputs > 0.5).float()
            corrects += (predicted == labels).sum().item()
            
        avg_loss = total_loss / total_samples
        accuracy = corrects / total_samples
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
# ...existing code...
