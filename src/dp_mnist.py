import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
import numpy as np
import multiprocessing as mp

# Enable cuDNN auto-tuner
cudnn.benchmark = True

# Model architecture
class DPCNN(nn.Module):
    def __init__(self):
        super(DPCNN, self).__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)  # Output: 16x28x28
        self.pool1 = nn.MaxPool2d(2)  # Output: 16x14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)  # Output: 32x14x14
        self.pool2 = nn.MaxPool2d(2)  # Output: 32x7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_mnist(batch_size=250):
    """Load MNIST dataset with parallel data loading."""
    # Use 7 cores for data loading (leave 1 for system)
    num_workers = mp.cpu_count() - 1
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def train_model(train_loader, test_loader, params=None):
    """Train static DP-SGD model with fixed noise."""
    if params is None:
        params = {
            'learning_rate': 0.15,
            'max_grad_norm': 1.0,
            'base_noise_multiplier': 1.1,
            'delta': 1e-5,
            'epochs': 15,
            'max_epsilon': 1.0  # Privacy budget limit
        }
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DPCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    
    # Setup privacy engine
    privacy_engine = PrivacyEngine(accountant='rdp')
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=params['base_noise_multiplier'],
        max_grad_norm=params['max_grad_norm']
    )
    
    # Initialize metrics tracking
    metrics = {
        'epsilon_history': [],
        'noise_levels': [],
        'gradient_norms': [],
        'accuracy_history': [],
        'train_loss': [],
        'test_loss': []
    }
    
    # Training loop
    for epoch in range(params['epochs']):
        model.train()
        epoch_loss = []
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            
            # Check privacy budget
            current_epsilon = privacy_engine.get_epsilon(delta=params['delta'])
            if current_epsilon > params['max_epsilon']:
                print(f"Privacy budget exceeded (ε = {current_epsilon:.2f}). Stopping training.")
                return metrics
            
            # Record gradient norms before clipping
            total_norm = torch.norm(
                torch.stack([
                    torch.norm(p.grad.detach(), 2)
                    for p in model.parameters()
                ]), 2
            )
            metrics['gradient_norms'].append(total_norm.item())
            
            optimizer.step()
            epoch_loss.append(loss.item())
        
        # Evaluate
        model.eval()
        correct = 0
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += nn.functional.cross_entropy(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        accuracy = correct / len(test_loader.dataset)
        
        # Update metrics
        metrics['epsilon_history'].append(
            privacy_engine.get_epsilon(delta=params['delta']))
        metrics['noise_levels'].append(params['base_noise_multiplier'])
        metrics['accuracy_history'].append(accuracy)
        metrics['train_loss'].append(np.mean(epoch_loss))
        metrics['test_loss'].append(test_loss / len(test_loader))
        
        print(f'Static Model - Epoch {epoch+1}:')
        print(f'Train Loss: {metrics["train_loss"][-1]:.4f}')
        print(f'Test Loss: {metrics["test_loss"][-1]:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'ε = {metrics["epsilon_history"][-1]:.2f}')
    
    return metrics

def main():
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist()
    
    print("Training model...")
    metrics = train_model(train_loader, test_loader)

if __name__ == "__main__":
    print("Starting DP-SGD MNIST training with Rényi Differential Privacy...")
    main()
