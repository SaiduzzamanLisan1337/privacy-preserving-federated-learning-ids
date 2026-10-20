import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import load_data
from src.client import FLClient
from src.server import FLServer
from src.model import IDSModel


def train_centralized(input_dim, train_loader, test_loader, epochs=5):
    """Trains a baseline centralized model on all combined data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IDSModel(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def main():
    # Pipeline Configurations
    NUM_CLIENTS = 3
    COMMUNICATION_ROUNDS = 5
    LOCAL_EPOCHS = 2
    ALPHA = 0.5  # Dirichlet non-IID parameter (smaller = higher heterogeneity)
    TARGET_EPSILON = 10.0  # Differential privacy budget
    TARGET_DELTA = 1e-5  # Privacy failure probability threshold
    DATASET_PATH = 'data/nsl-kdd/KDDTrain+.txt'

    print("==================================================")
    print("  Federated IDS with Opacus DP-SGD & Dirichlet    ")
    print("==================================================")
    print(f"Loading NSL-KDD with Non-IID Partitioning (\u03b1 = {ALPHA})...\n")

    client_loaders, cent_train, test_loader, input_dim = load_data(
        filepath=DATASET_PATH,
        num_clients=NUM_CLIENTS,
        alpha=ALPHA
    )

    print("\n--- Centralized Training Baseline ---")
    cent_acc = train_centralized(input_dim, cent_train, test_loader, epochs=5)
    print(f"Centralized Model Accuracy: {cent_acc:.2f}%\n")

    print("--- Federated Learning (FedAvg) with DP-SGD ---")
    server = FLServer(input_dim)
    clients = [FLClient(i, client_loaders[i]) for i in range(NUM_CLIENTS)]

    for round_idx in range(1, COMMUNICATION_ROUNDS + 1):
        print(f"\n--- Communication Round {round_idx}/{COMMUNICATION_ROUNDS} ---")
        client_weights = []

        for client in clients:
            weights = client.train(
                global_model=server.global_model,
                local_epochs=LOCAL_EPOCHS,
                target_epsilon=TARGET_EPSILON,
                target_delta=TARGET_DELTA
            )
            client_weights.append(weights)

        server.aggregate_fedavg(client_weights)
        acc = server.evaluate(test_loader)
        print(f"Round {round_idx}/{COMMUNICATION_ROUNDS} - Global Model Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()