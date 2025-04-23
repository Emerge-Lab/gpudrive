import torch

from gpudrive.networks.late_fusion import NeuralNet


def load_policy(path_to_cpt, model_name, device, env_config=None):
    """Load a policy from a given path."""

    # Load a trained model
    saved_cpt = torch.load(
        f=f"{path_to_cpt}/{model_name}.pt",
        map_location=device,
        weights_only=False,
    )

    print(f"Load model from {path_to_cpt}/{model_name}.pt")

    # Create policy architecture from saved checkpoint
    policy = NeuralNet(
        input_dim=saved_cpt["model_arch"]["input_dim"],
        action_dim=saved_cpt["action_dim"],
        hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
        config=env_config,
    ).to(device)

    # Load the model parameters
    policy.load_state_dict(saved_cpt["parameters"])

    return policy.eval()
