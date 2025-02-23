from gpudrive.networks.late_fusion import NeuralNet

if __name__ == "__main__":

    # Load pre-trained policy from the Hub
    policy = NeuralNet.from_pretrained("daphne-cornelisse/policy_test")
    
