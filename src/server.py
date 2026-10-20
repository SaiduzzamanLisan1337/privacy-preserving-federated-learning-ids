import torch
from src.model import IDSModel

class FLServer:
    def __init__(self, input_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = IDSModel(input_dim).to(self.device)
        
    def aggregate_fedavg(self, client_weights):
        global_dict = self.global_model.state_dict()
        
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_weights[i][k].float() for i in range(len(client_weights))], 0).mean(0)
            
        self.global_model.load_state_dict(global_dict)
        
    def evaluate(self, dataloader):
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for features, labels in dataloader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.global_model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
