import torch
import torch.nn as nn
import torch.optim as optim
import copy
from opacus import PrivacyEngine

class FLClient:
    def __init__(self, client_id, dataloader):
        self.client_id = client_id
        self.dataloader = dataloader
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, global_model, local_epochs, lr=0.01, target_epsilon=10.0, target_delta=1e-5):
        model = copy.deepcopy(global_model).to(self.device)
        model.train()  # Explicitly set training mode before Opacus validation
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        privacy_engine = PrivacyEngine()
        
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=self.dataloader,
            epochs=local_epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=1.2,
        )
        
        for epoch in range(local_epochs):
            for features, labels in dataloader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                loss = self.criterion(model(features), labels)
                loss.backward()
                optimizer.step()
                
        epsilon = privacy_engine.get_epsilon(target_delta)
        print(f"[Client {self.client_id}] DP Training completed. \u03b5 = {epsilon:.2f}, \u03b4 = {target_delta}")
        
        return model._module.state_dict()
