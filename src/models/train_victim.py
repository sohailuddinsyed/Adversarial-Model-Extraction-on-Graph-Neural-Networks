import torch
import torch.nn.functional as F
from src.models.gcn import GCN
from src.data.dataset_loader import DatasetLoader
import os


class VictimTrainer:
    """Train victim GCN model on citation networks."""
    
    def __init__(self, dataset_name, hidden_dim=16, lr=0.01, weight_decay=5e-4):
        """
        Initialize victim trainer.
        
        Args:
            dataset_name: 'Cora' or 'Pubmed'
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.dataset_name = dataset_name
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Load dataset
        self.loader = DatasetLoader(dataset_name)
        self.data = self.loader.load()
        
        # Initialize model
        self.model = GCN(
            num_features=self.loader.dataset.num_features,
            hidden_dim=hidden_dim,
            num_classes=self.loader.dataset.num_classes
        )
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
    def train_epoch(self, train_mask):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.nll_loss(out[train_mask], self.data.y[train_mask])
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, mask):
        """Evaluate model on given mask."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred[mask] == self.data.y[mask]).sum()
            acc = int(correct) / int(mask.sum())
        return acc
    
    def train(self, num_train_per_class, epochs=200, verbose=True):
        """
        Train victim model.
        
        Args:
            num_train_per_class: Training samples per class (140 for Cora, 60 for Pubmed)
            epochs: Number of training epochs
            verbose: Print training progress
            
        Returns:
            Training history
        """
        train_mask = self.loader.get_train_mask(num_train_per_class)
        
        history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
        
        for epoch in range(epochs):
            loss = self.train_epoch(train_mask)
            train_acc = self.evaluate(train_mask)
            test_acc = self.evaluate(self.data.test_mask)
            
            history['train_loss'].append(loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1:03d}: Loss={loss:.4f}, '
                      f'Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}')
        
        return history
    
    def save_model(self, path):
        """Save trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'dataset_name': self.dataset_name,
            'hidden_dim': self.hidden_dim,
            'num_features': self.loader.dataset.num_features,
            'num_classes': self.loader.dataset.num_classes
        }, path)
        print(f'Model saved to {path}')
    
    def load_model(self, path):
        """Load trained model."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Model loaded from {path}')
