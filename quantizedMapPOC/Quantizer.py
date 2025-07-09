import torch
import torch.nn as nn
import torch.nn.functional as F

def scale_mapper_func(x: torch.Tensor, const: float = 1.0) -> torch.Tensor:
    """
    Scale and offset the input tensor x using min-max normalization.
    """
    x_max = x.max().item()
    x_min = x.min().item()
    return const * (x-x_min) / (x_max - x_min)

def inv_scale_mapper_func(x: torch.Tensor, xmax: float, xmin: float, const: float = 1.0) -> torch.Tensor:
    """
    Inverse scale and offset the input tensor x to the original float range.
    """
    return (x * (xmax - xmin))/const + xmin

class Standardizer:
    def __init__(self, Qmin=-128, Qmax=127):
        self.mean = None
        self.std = None
        self.Qmin = Qmin
        self.Qmax = Qmax
        self.const = 1.0

    def fit(self, x: torch.Tensor):
        self.mean = torch.mean(x)
        self.std = torch.std(x, unbiased=False)
        if self.std == 0.0 or torch.isnan(self.std):
            self.std = 1.0
            

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer not fitted yet. Call 'fit' with training data.")
        if x.numel() == 0:
            raise ValueError("Input tensor is empty.")
        x_norm = (x - self.mean) / self.std
        x_min = x_norm.min().item()
        x_max = x_norm.max().item()
        if x_min < 0 and x_max > 0:
            self.const = min(self.Qmax/(x_max), self.Qmin/(x_min))
        elif x_min >= 0:
            if x_max != 0:
                self.const = self.Qmax/(x_max)
        else:  # x_max <= 0
            if x_min != 0:
                self.const = self.Qmin/(x_min)
        return x_norm * self.const

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer not fitted yet. Call 'fit' with training data.")
        return (x/self.const) * self.std + self.mean

def softmax_contrastive_loss(x: torch.Tensor, x_hat: torch.Tensor, temperature: float = 1.0):
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if x_hat.dim() == 1:
        x_hat = x_hat.unsqueeze(1)
    dists = torch.cdist(x_hat, x, p=2) ** 2  # shape (N, N)
    n = dists.shape[0]
    scaled_exp = torch.exp(dists / temperature)
    numerator = torch.diagonal(scaled_exp)
    denominator = (torch.sum(scaled_exp, dim=1) - torch.diagonal(scaled_exp))
    loss = torch.log(numerator / denominator)
    return torch.mean(loss)

def train(x: torch.Tensor, c: torch.nn.Parameter, d: torch.nn.Parameter, Qmin=-128, Qmax=127, train_logs: bool = True):
    # Optimizer
    optimizer = torch.optim.Adam([c, d], lr=0.01)

    # Training loop to optimize c and d
    for step in range(1000):
        optimizer.zero_grad()
        if c.item() != 0.0:
            x_q = torch.round(x/c - d).clamp(Qmin, Qmax)
        else:
            x_q = torch.tensor([Qmax]*len(x), device=x.device)
        x_hat = c * (x_q + d)

        # Contrastive loss
        # loss = softmax_contrastive_loss(x, x_hat)

        # Standard MSE loss
        loss = F.mse_loss(x_hat, x)

        loss.backward()
        optimizer.step()
        if train_logs:
            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss.item()}, c = {c.item()}, d = {d.item()}")
                # print("Quantized values:", x_q)
                # print("Dequantized values:", x_hat)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Sample data (your float list)
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32, device=device)
    x = x/7.0
    x_comp = x + 0.0002

    # Scale the input tensor
    x_scale = scale_mapper_func(x, const=7.0)

    # Bounds of float values
    fmin_org, fmax_org = x.min().item(), x.max().item()

    # Quantization bounds (e.g., int8: 0 to 255 or -128 to 127)
    Qmin, Qmax = -128, 127

    # Calculate the min and max of the scale mapped float values
    fmin, fmax = x_scale.min().item(), x_scale.max().item()

    # Parameters: scale (c) and offset (d)
    # Initialize as learnable parameters
    c = torch.nn.Parameter(torch.tensor((fmax_org-fmin_org)/(Qmax-Qmin), device=device))  # scale
    d = torch.nn.Parameter(torch.tensor((fmax_org*Qmin-fmin_org*Qmax)/(Qmax-Qmin), device=device))  # zero-point

    # Cheat Inits
    # c = torch.nn.Parameter(torch.tensor(0.01, device=device))
    # d = torch.nn.Parameter(torch.tensor(0.02, device=device))

    print("Original values:", x)
    print("Scaled values:", x_scale)

    # Strategy 1: Without initial scaling
    print("Strategy 1: Without initial scaling")
    train(x, c, d, train_logs=True)
    x_q_final = torch.round((x - d) / c).clamp(Qmin, Qmax)
    x_hat_final = c * x_q_final + d

    print("Final quantized values:", x_q_final)
    print("Final dequantized values:", x_hat_final)
    print("\n\n\n")

    # Strategy 2: With initial scaling
    print("Strategy 2: With initial scaling")
    c = torch.nn.Parameter(torch.tensor((fmax-fmin)/(Qmax-Qmin), device=device))  # scale
    d = torch.nn.Parameter(torch.tensor((fmax*Qmin-fmin*Qmax)/(Qmax-Qmin), device=device))  # zero-point
    train(x_scale, c, d, train_logs=True)
    x_q_final = torch.round((x_scale - d) / c).clamp(Qmin, Qmax)
    x_hat_final = c * x_q_final + d

    print("Scaled values:", x_scale)
    print("Final quantized values:", x_q_final)
    print("Final dequantized values:", inv_scale_mapper_func(x_hat_final, fmax_org, fmin_org, const=7.0))
    print("\n\n\n")
    # print(softmax_contrastive_loss(x, x_comp).item())
    # x_neg_comp = x_comp.clone()
    # x_neg_comp[0] = -x_neg_comp[0]
    # print(f'Contrastive loss for {x_neg_comp} = {softmax_contrastive_loss(x, x_neg_comp)}')

    # print(f'MSE loss for {x_comp} = {F.mse_loss(x, x_comp).item()}')
    # print(f'MSE loss for {x_neg_comp} = {F.mse_loss(x, x_neg_comp).item()}')