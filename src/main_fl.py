import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import load_data
from src.client import FLClient
from src.server import FLServer
from src.model import IDSModel

def train_centralized(input_dim, train_loader, test_loader, epochs=10):
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
            _, predicted = torch.max(model(features).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    NUM_CLIENTS = 3
    COMMUNICATION_ROUNDS = 5
    LOCAL_EPOCHS = 2
    
    print("Loading NSL-KDD dataset...")
    client_loaders, cent_train, test_loader, input_dim = load_data('data/nsl-kdd/KDDTrain+.txt', num_clients=NUM_CLIENTS)
    
    print(f"\n--- Centralized Training Baseline ---")
    cent_acc = train_centralized(input_dim, cent_train, test_loader, epochs=5)
    print(f"Centralized Model Accuracy: {cent_acc:.2f}%\n")
    
    print(f"--- Federated Learning (FedAvg) with DP-SGD ---")
    server = FLServer(input_dim)
    clients = [FLClient(i, client_loaders[i]) for i in range(NUM_CLIENTS)]
    
    for round_idx in range(COMMUNICATION_ROUNDS):
        client_weights = []
        for client in clients:
            weights = client.train(server.global_model, LOCAL_EPOCHS)
            client_weights.append(weights)
            
        server.aggregate_fedavg(client_weights)
        acc = server.evaluate(test_loader)
        print(f"Round {round_idx + 1}/{COMMUNICATION_ROUNDS} - Global Model Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
