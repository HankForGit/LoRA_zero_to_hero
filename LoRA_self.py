import torch
import torch.nn as nn
import torch.nn.functional as F

#seed
torch.manual_seed(42)
#simple LoRA layer
class LoRA_linear(nn.Module):
    def __init__(self,base_layer,rank=2,alpha=2):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha/rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        for parameter in base_layer.parameters():
            parameter.requires_grad = False

        self.A = nn.Parameter(
            torch.randn(rank,in_features) * 0.01
        )
        self.B = nn.Parameter(
            torch.zeros(out_features,rank)
        )
    def forward(self,x):
        base_output = self.base(x)
        lora_output = F.linear(x,(self.B @ self.A))
        return base_output + self.scale * lora_output
    def merged_layer(self):
        return self.base.weight + self.scale * (self.B @ self.A)

#creating answer

input_size = 8
output_size = 6
rank = 2
lora_alpha = 2

base_layer = nn.Linear(input_size,output_size)

original_weight = base_layer.weight.detach().clone()

true_A = nn.Parameter(
    torch.randn(rank,input_size) * 0.5
)
true_B = nn.Parameter(
    torch.randn(output_size,rank)
)
target_weight = base_layer.weight + true_B @ true_A
num_sample = 1024
x_train = nn.Parameter(
    torch.randn(num_sample,input_size)
)
with torch.no_grad():
    y_train = F.linear(x_train,target_weight,base_layer.bias)

#model and optimizer
model = LoRA_linear(
    base_layer = base_layer,
    rank = rank,
    alpha = lora_alpha
)
optimizer = torch.optim.Adam(
    [model.A,model.B],
    lr = 0.05
)

#prediction and loss before training
with torch.no_grad():
    prediction_before = model(x_train)
    loss_before = F.mse_loss(prediction_before,y_train)
    print(f"Loss before training:{loss_before.item():.8f}")
#training
for step in range(1000):
    prediction = model(x_train)
    loss = F.mse_loss(prediction,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step:{step} | Loss:{loss.item():.8f}")
#after training
with torch.no_grad():
    prediction_after = model(x_train)
    loss_after = F.mse_loss(prediction_after,y_train)
    print(f"Loss after training:{loss_after.item():.8f}")
#make sure the original weight is frozen
assert torch.allclose(model.base.weight, original_weight)

#parameters and trainable parameters
full_weight_parameters = input_size * output_size
lora_parameters = model.A.numel() + model.B.numel()

print(f"Full weight parameter:{full_weight_parameters}")
print(f"LoRA parameter:{lora_parameters}")
print("Trainable parameter:")
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(f"{name}: {tuple(parameter.shape)}")

#save and merge
adapter = {
    "A": model.A,
    "B": model.B,
    "rank" : rank,
    "alpha": lora_alpha
}
torch.save(adapter,"lora_adapter.safetensors")
print("Adapter saved to lora_adapter.safetensors")
with torch.no_grad():
    merged_layer = nn.Linear(input_size,output_size)
    merged_layer.weight.copy_(model.merged_layer())
    merged_layer.bias.copy_(model.base.bias)


    merged_output = merged_layer(x_train)
    lora_output = model(x_train)
    max_diff = (lora_output - merged_output).abs().max()

print(f"Maximum difference between training model and merged model is {max_diff}")
