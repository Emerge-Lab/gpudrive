import torch
from huggingface_hub import login
from gpudrive.utils.config import load_config
from gpudrive.networks.late_fusion import NeuralNet

login()

config = load_config("examples/experimental/config/reliable_agents_params")
checkpoint_path = "examples/experimental/models/model_PPO____R_10000__02_27_09_19_10_626_003200.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saved_cpt = torch.load(
    checkpoint_path, map_location=device, weights_only=False
)

# Reconstruct the model
policy = NeuralNet(
    input_dim=saved_cpt["model_arch"]["input_dim"],
    action_dim=saved_cpt["action_dim"],
    hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
    config=config,
).to(device)

# Load the model parameters
policy.load_state_dict(saved_cpt["parameters"])

# Set model to eval mode
policy.eval()

# policy.save_pretrained("policy_test")  # Saves both the model weights and config.json

# Push to Hugging Face Hub
print("Pushing model to Hugging Face...")
policy.push_to_hub("daphne-cornelisse/policy_S10_000_02_27")
