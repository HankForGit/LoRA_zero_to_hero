# LoRA_zero_to_hero
# Minimal LoRA Implementation in PyTorch

This project is a small implementation of LoRA (Low-Rank Adaptation) using only PyTorch.

The goal is to understand the main idea behind LoRA without using Hugging Face PEFT or other LoRA libraries.

The program:

- Creates a basic linear layer
- Freezes the original weights
- Adds two small trainable matrices
- Trains only the new matrices
- Saves the trained LoRA adapter
- Merges the adapter into the original layer
- Checks that merged and unmerged models give the same output

## What Is LoRA?

A normal linear layer uses a weight matrix:

```text
W
```

During full fine-tuning, every value in `W` can be updated.

LoRA keeps `W` frozen and learns two smaller matrices:

```text
A and B
```

The weight update is:

```text
delta_W = B @ A
```

The final weight is:

```text
W_new = W + (alpha / rank) * delta_W
```

The model output is therefore:

```text
output = base_output + (alpha / rank) * lora_output
```

Where:

- `W` is the frozen base weight
- `A` and `B` are trainable LoRA matrices
- `rank` controls the size of the LoRA matrices
- `alpha / rank` controls the strength of the LoRA update

## Why Use LoRA?

For a linear layer with:

```text
input size  = d_in
output size = d_out
```

The full weight contains:

```text
d_in * d_out
```

parameters.

LoRA only trains:

```text
rank * (d_in + d_out)
```

parameters.

When the model is large and the rank is small, this can greatly reduce the number of trainable parameters.

## Project Structure

```text
minimal-lora/
├── LoRAfromscratch.py
├── README.md
├── requirements.txt
└── .gitignore
```

After running the program, it will also create:

```text
micro_lora_adapter.pt
```

This file stores the trained LoRA matrices and settings.

## Requirements

- Python 3.10 or newer
- PyTorch

Install PyTorch with:

```bash
pip install torch
```

Or install the project requirements:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
torch
```

## How to Run

Clone the repository:

```bash
git clone <your-repository-url>
cd minimal-lora
```

Run the program:

```bash
python LoRAfromscratch.py
```

## What the Program Does

### 1. Creates a base linear layer

```python
base_layer = nn.Linear(input_size, output_size)
```

This layer represents a small pre-trained model.

### 2. Creates a target task

The program makes a new target weight by adding a low-rank change to the original weight:

```python
target_weight = base_weight + true_delta_weight
```

This creates a simple task that LoRA can learn.

### 3. Freezes the base layer

```python
for parameter in self.base.parameters():
    parameter.requires_grad = False
```

The optimizer cannot update the original weight or bias.

### 4. Creates the LoRA matrices

```python
self.A = nn.Parameter(
    torch.randn(rank, in_features) * 0.01
)

self.B = nn.Parameter(
    torch.zeros(out_features, rank)
)
```

Their shapes are:

```text
A: [rank, input size]
B: [output size, rank]
```

Therefore:

```text
B @ A: [output size, input size]
```

This is the same shape as the original weight.

### 5. Adds the LoRA output

```python
delta_weight = self.B @ self.A
lora_output = F.linear(x, delta_weight)

return base_output + self.scale * lora_output
```

The base layer still produces its normal output. The LoRA update is added to it.

### 6. Trains only A and B

```python
optimizer = torch.optim.Adam(
    [model.A, model.B],
    lr=0.05
)
```

The optimizer only receives `A` and `B`, so it does not update the base layer.

### 7. Checks that the base weight stayed frozen

The program saves a copy of the base weight before training and compares it after training:

```python
assert torch.allclose(
    base_layer.weight,
    original_weight
)
```

If the base weight changed, this check would fail.

### 8. Saves the adapter

The program saves only the LoRA matrices and settings:

```python
adapter = {
    "A": model.A.detach().cpu(),
    "B": model.B.detach().cpu(),
    "rank": model.rank,
    "alpha": model.alpha
}
```

This is smaller than saving another full copy of the model.

### 9. Merges LoRA into the base layer

For inference, the LoRA update can be added directly to the base weight:

```python
merged_weight = base_weight + scale * (B @ A)
```

After merging, a normal linear layer can be used without a separate LoRA calculation.

The program compares the outputs from the merged and unmerged models. Their difference should be very small because of normal floating-point rounding.

## Expected Result

The exact numbers may be different, but the output should show:

```text
Loss before training: ...
Step    0 | Loss: ...
Step  100 | Loss: ...
...
Loss after training: ...

Base weight remained frozen: True

Full weight parameters: 48
LoRA trainable parameters: 28

Saved adapter to micro_lora_adapter.pt
Maximum difference after merging: ...
```

The important results are:

- The loss becomes much smaller
- The base weight stays unchanged
- Only `A` and `B` are trainable
- The adapter is saved
- Merged and unmerged outputs are almost the same

## Why Is A Random and B Zero?

The program starts with:

```python
A = small random values
B = zeros
```

Because `B` starts at zero:

```text
B @ A = 0
```

The LoRA layer does not change the base model at the start of training.

Both matrices should not start at zero. If both are zero, they may not receive useful gradients at the beginning of training.

## Rank and Alpha

### Rank

The rank controls the size of the LoRA matrices.

A smaller rank:

- Uses fewer parameters
- Uses less memory
- May not learn a complex weight change

A larger rank:

- Uses more parameters
- Can learn more complex changes
- Reduces some of LoRA's parameter savings

### Alpha

Alpha controls the size of the LoRA update through:

```text
scale = alpha / rank
```

In this example, `alpha` can be set equal to `rank`:

```python
rank = 2
alpha = 2
```

This gives:

```text
scale = 2 / 2 = 1
```

Setting `alpha` equal to `rank` is only a simple choice for this example. It is not required by LoRA.

## Experiments

You can change the code to test your understanding.

### Experiment 1: Change the rank

Try:

```python
rank = 1
```

Then try:

```python
rank = 4
```

Compare:

- Final loss
- Number of trainable parameters
- Ability to learn the target weight change

### Experiment 2: Change alpha

Try:

```python
rank = 2
alpha = 4
```

The scale will become:

```text
4 / 2 = 2
```

Observe how the scale affects training.

### Experiment 3: Print the gradients

After `loss.backward()`, add:

```python
print(model.A.grad)
print(model.B.grad)
print(model.base.weight.grad)
```

`A` and `B` should receive gradients. The frozen base weight should not receive a gradient.

### Experiment 4: Remove the freezing step

Temporarily remove:

```python
parameter.requires_grad = False
```

Then check which parameters can receive gradients.

### Experiment 5: Compare merged outputs

Change the input data and confirm that the merged and unmerged layers still produce almost the same output.

## Limits of This Project

This project is made for learning. It is not a full LoRA training library.

It does not include:

- A Transformer model
- Real language-model training
- LoRA dropout
- Quantization
- Multiple adapters
- Automatic target-layer selection
- Mixed-precision training
- Distributed training
- GPU memory optimization

Libraries such as Hugging Face PEFT provide these features for real model training.

## What I Learned

Through this project, I learned how to:

- Build a custom PyTorch module
- Freeze model parameters
- Control which parameters an optimizer updates
- Use matrix multiplication in a LoRA layer
- Understand the role of rank and alpha
- Check whether gradients and parameter updates are correct
- Save LoRA adapter parameters
- Merge an adapter into a base weight
- Compare merged and unmerged model outputs

## Note

This project is an educational implementation. It was created to understand the basic LoRA method through a small and readable example.
