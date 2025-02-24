import torch
from gpudrive.networks.late_fusion import NeuralNet 
from huggingface_hub import login
from transformers import PretrainedConfig

# Hugging Face login 
login()

# Define the checkpoint path
checkpoint_path = "examples/experimental/eval/models/model_PPO____R_10000__02_23_13_59_35_797_000500.pt"

# Load the checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saved_cpt = torch.load(checkpoint_path, map_location=device)

# Reconstruct the model
policy = NeuralNet(
    input_dim=saved_cpt["model_arch"]["input_dim"],
    action_dim=saved_cpt["action_dim"],
    hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
).to(device)

# Load the model parameters
policy.load_state_dict(saved_cpt["parameters"])

# Set model to eval mode
policy.eval()

# Create the configuration (for Hugging Face)
class CustomConfig(PretrainedConfig):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

# Create a config object with the architecture parameters
config = CustomConfig(
    input_dim=saved_cpt["model_arch"]["input_dim"],
    action_dim=saved_cpt["action_dim"],
    hidden_dim=saved_cpt["model_arch"]["hidden_dim"]
)

# Save model and config together
policy.config = config
#policy.save_pretrained("policy_test")  # Saves both the model weights and config.json

# Push to Hugging Face Hub
print('Pushing model to Hugging Face...')
policy.push_to_hub("daphne-cornelisse/policy_S10_000")
