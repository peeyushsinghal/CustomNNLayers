"""
CNN Model for MNIST using custom layers with autograd
"""
# Third Party Imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Local Imports
from custom_layers import conv2d, relu, max_pool2d


class CustomCNN(torch.nn.Module):
    """
    Custom CNN Model for MNIST using custom layers with autograd
    """
    def __init__(self):
        """
        Initialize the CustomCNN model
        """
        super(CustomCNN, self).__init__()

        # Initialize learnable parameters
        self.conv1_kernel = torch.nn.Parameter(torch.randn(5, 5) * 0.01)
        self.conv2_kernel = torch.nn.Parameter(torch.randn(5, 5) * 0.01)

        # Update the input size for fc1 based on the actual flattened size
        self.fc1 = torch.nn.Linear(49, 128)  # Changed from 392 to 49 (7*7)
        self.fc2 = torch.nn.Linear(128, 10)
    
    def forward(self, x):
        """
        Forward pass of the CustomCNN model
        """
        batch_size = x.size(0)
        # MNIST images are (batch_size, 1, 28, 28)
        x = x.view(batch_size, 28, 28)  # Reshape to (batch_size, height, width)
        outputs = []
        for i in range(batch_size):
            # Process each image in the batch separately
            xi = x[i]  # Get single image
            xi = conv2d(xi, self.conv1_kernel, 2, 1)  # Output: 24x24
            xi = relu(xi)
            xi = max_pool2d(xi)  # Output: 12x12
            
            # Second conv block
            xi = conv2d(xi, self.conv2_kernel, 2, 1)  # Output: 8x8
            xi = relu(xi)
            xi = max_pool2d(xi)  # Output: 7x7
            
            # Store processed image
            outputs.append(xi)
        
        # Stack processed images back into batch
        x = torch.stack(outputs)
        
        # Flatten and dense layers
        x = x.view(batch_size, -1)  # Will be (batch_size, 49)
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


def train_model(model, train_loader, optimizer, epochs=5):
    """
    Train the CustomCNN model
    """
    model.train()
    criterion = torch.nn.NLLLoss()
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def evaluate_model(model, test_loader):
    """
    Evaluate the CustomCNN model
    """
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.NLLLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy


def main():
    """
    Main function to train and evaluate the CustomCNN model
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True)
    
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)
    
    # Create model and optimizer
    model = CustomCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train and evaluate
    for epoch in range(1):
        print(f"\nEpoch {epoch+1}/5")
        print("-" * 60)
        train_model(model, train_loader, optimizer)
        accuracy = evaluate_model(model, test_loader)
        print(f"Epoch {epoch+1} Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()