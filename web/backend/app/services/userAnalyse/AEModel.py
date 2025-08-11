import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class V4Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.cover_adapter = MultiScaleFusion()

        self.encoder = nn.Sequential(
            nn.Linear(42, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 42),
        )

    def forward(self, inputs):
        user_feature = inputs["user_feature"]  # [B,21]
        cover_feature = inputs["cover_feature"]  # [B,2048]

        # 多尺度融合
        fused_cover = self.cover_adapter(cover_feature)  # [B,21]

        x = self.encoder(torch.cat([user_feature, fused_cover], dim=1))
        x = self.decoder(x)
        return x


class MultiScaleFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(0.2),  # 新增正则化
        )
        self.branch2 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.GELU(),
            nn.Dropout(0.2),  # 新增正则化
        )
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 21),  # 直接映射到21维
            nn.LayerNorm(21),
        )
        self.attention = ChannelAttention(512 + 256)  # 新增注意力

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        fused = torch.cat([x1, x2], dim=1)
        fused = self.attention(fused)  # 应用注意力
        return self.fusion(fused)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size()
        # print(x.size())
        att = self.gap(x.view(b, c, 1))  # [B, C, 1]
        att = self.fc(att.view(b, c)).view(b, c, 1)
        return x * att.expand_as(x.view(b, c, 1)).view(b, c)


class ReconstructionLoss(nn.Module):
    def __init__(self, user_weight=0.7):
        super().__init__()
        self.mse = nn.MSELoss()
        self.user_weight = user_weight  # 用户特征重建权重

    def forward(self, outputs, inputs, model):
        # 动态计算输入特征组合
        with torch.no_grad():
            fused_cover = model.cover_adapter(inputs["cover_feature"])
            input_features = torch.cat([inputs["user_feature"], fused_cover], dim=1)

        # 分离用户特征和封面特征重建
        user_loss = self.mse(outputs[:, :21], input_features[:, :21])
        cover_loss = self.mse(outputs[:, 21:], input_features[:, 21:])

        return self.user_weight * user_loss + (1 - self.user_weight) * cover_loss


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(residual)
        return F.relu(self.block(x) + residual)


def infer_loss(user_feature, cover_feature):
    model = V4Model().to(device)
    model.load_state_dict(torch.load("app/services/userAnalyse/best_model.pth", map_location=device))
    model.eval()

    loss_fn = ReconstructionLoss()
    total_error = 0.0

    with torch.no_grad():
        input = {
            "user_feature": user_feature.to(device),
            "cover_feature": cover_feature.to(device),
        }

        # Forward pass
        output = model(input)

        # Calculate reconstruction error
        error = loss_fn(output, input, model)
        total_error = error.item()

    # Calculate average reconstruction error
    avg_error = total_error
    return avg_error


def infer_8features(user_feature, cover_feature):
    model = V4Model().to(device)
    model.load_state_dict(torch.load("app/services/userAnalyse/best_model.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        user_feature = user_feature.to(device)
        cover_feature = cover_feature.to(device)

        fused_cover = model.cover_adapter(cover_feature)

        feature8 = model.encoder(torch.cat([user_feature, fused_cover], dim=1))

    return feature8
