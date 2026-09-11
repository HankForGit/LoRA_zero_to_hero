import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)


# ---------------------------------------------------------
# 1. 一个最小的 LoRA Linear 层
# ---------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=2, alpha=2):
        super().__init__()

        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # 冻结预训练层 W 和 bias
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        # A: [rank, in_features]
        # B: [out_features, rank]
        self.A = nn.Parameter(
            torch.randn(rank, in_features) * 0.01
        )
        self.B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

    def forward(self, x):
        # 原始模型输出：x W^T + bias
        base_output = self.base(x)

        # LoRA 权重更新：delta_W = B @ A
        delta_weight = self.B @ self.A

        # F.linear 内部计算 x @ delta_weight.T
        lora_output = F.linear(x, delta_weight)

        return base_output + self.scale * lora_output

    def merged_weight(self):
        """
        将 LoRA 更新合并进原始权重。
        推理时可以不再单独保留 LoRA 计算。
        """
        return self.base.weight + self.scale * (self.B @ self.A)


# ---------------------------------------------------------
# 2. 创建一个“预训练”线性层
# ---------------------------------------------------------

input_size = 8
output_size = 6
rank = 2
lora_alpha = 2

base_layer = nn.Linear(input_size, output_size)

# 保存原始权重，用于确认训练时没有改变它
original_weight = base_layer.weight.detach().clone()


# ---------------------------------------------------------
# 3. 创建一个新任务
#
# 假设新任务的理想权重是：
#
# target_W = original_W + low_rank_change
#
# 其中 low_rank_change 本身就是一个 rank=2 的矩阵。
# 因此 rank=2 的 LoRA 理论上可以学会它。
# ---------------------------------------------------------

true_A = torch.randn(rank, input_size)
true_B = torch.randn(output_size, rank)

true_delta_weight = 0.5 * (true_B @ true_A)
target_weight = base_layer.weight.detach() + true_delta_weight

# 生成模拟训练数据
num_samples = 1024

x_train = torch.randn(num_samples, input_size)

with torch.no_grad():
    y_train = F.linear(
        x_train,
        target_weight,
        base_layer.bias
    )


# ---------------------------------------------------------
# 4. 将原始层包装成 LoRA 层
# ---------------------------------------------------------

model = LoRALinear(
    base_layer=base_layer,
    rank=rank,
    alpha=lora_alpha
)

# 优化器只会收到 A 和 B，因为原始层已经冻结
optimizer = torch.optim.Adam(
    [model.A, model.B],
    lr=0.05
)


# ---------------------------------------------------------
# 5. 查看训练前的误差
# ---------------------------------------------------------

with torch.no_grad():
    prediction_before = model(x_train)
    loss_before = F.mse_loss(prediction_before, y_train)

print(f"Loss before training: {loss_before.item():.6f}")


# ---------------------------------------------------------
# 6. 训练 LoRA
# ---------------------------------------------------------

for step in range(1000):
    prediction = model(x_train)
    loss = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.8f}")


# ---------------------------------------------------------
# 7. 检查训练结果
# ---------------------------------------------------------

with torch.no_grad():
    prediction_after = model(x_train)
    loss_after = F.mse_loss(prediction_after, y_train)

print(f"\nLoss after training: {loss_after.item():.8f}")


# 原始权重必须保持不变
assert torch.allclose(
    base_layer.weight,
    original_weight
)

print("Base weight remained frozen: True")


# ---------------------------------------------------------
# 8. 统计参数量
# ---------------------------------------------------------

full_weight_parameters = input_size * output_size
lora_parameters = model.A.numel() + model.B.numel()

print(f"\nFull weight parameters: {full_weight_parameters}")
print(f"LoRA trainable parameters: {lora_parameters}")

print("\nTrainable parameters:")

for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(f"  {name}: {tuple(parameter.shape)}")


# ---------------------------------------------------------
# 9. 保存 LoRA adapter
#
# 不需要保存完整模型，只保存 A、B 和配置即可。
# ---------------------------------------------------------

adapter = {
    "A": model.A.detach().cpu(),
    "B": model.B.detach().cpu(),
    "rank": model.rank,
    "alpha": model.alpha
}

torch.save(adapter, "micro_lora_adapter.pt")

print("\nSaved adapter to micro_lora_adapter.pt")


# ---------------------------------------------------------
# 10. 将 LoRA 合并回普通 Linear 层
# ---------------------------------------------------------

merged_layer = nn.Linear(input_size, output_size)

with torch.no_grad():
    merged_layer.weight.copy_(model.merged_weight())
    merged_layer.bias.copy_(model.base.bias)

    lora_output = model(x_train)
    merged_output = merged_layer(x_train)

    max_difference = (
        lora_output - merged_output
    ).abs().max()

print(
    "Maximum difference after merging:",
    max_difference.item()
)