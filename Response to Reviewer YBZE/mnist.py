import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input images
        out1 = torch.relu(self.fc1(x))
        out2 = torch.relu(self.fc2(out1))
        out3 = torch.relu(self.fc3(out2))
        out4 = torch.relu(self.fc4(out3))
        logits = self.classifier(out4)
        return out1, out2, out3, out4, logits

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Set hidden layer dimensions
hidden_dim = 128

# Create the MLP model instance
model = MLP(hidden_dim).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    total_correct = [0] * 5
    total_samples = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        out1, out2, out3, out4, outputs = model(images)

        # model.classifier(out1)

        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy for each layer
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct[4] += (predicted == labels).sum().item()

        _, predicted = torch.max(model.classifier(out4).data, 1)
        total_correct[3] += (predicted == labels).sum().item()

        _, predicted = torch.max(model.classifier(out3).data, 1)
        total_correct[2] += (predicted == labels).sum().item()

        _, predicted = torch.max(model.classifier(out2).data, 1)
        total_correct[1] += (predicted == labels).sum().item()

        _, predicted = torch.max(model.classifier(out1).data, 1)
        total_correct[0] += (predicted == labels).sum().item()

    # Print the average loss for each epoch
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

    # Print accuracy for each layer
    for i, correct in enumerate(total_correct):
        accuracy = correct / total_samples
        print(f"Layer {i+1} Accuracy: {accuracy}")

    # Evaluation on the test set
    model.eval()
    correct = [0] * 5
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            out1, out2, out3, out4, outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct[4] += (predicted == labels).sum().item()

            _, predicted = torch.max(model.classifier(out4).data, 1)
            correct[3] += (predicted == labels).sum().item()

            _, predicted = torch.max(model.classifier(out3).data, 1)
            correct[2] += (predicted == labels).sum().item()

            _, predicted = torch.max(model.classifier(out2).data, 1)
            correct[1] += (predicted == labels).sum().item()

            _, predicted = torch.max(model.classifier(out1).data, 1)
            correct[0] += (predicted == labels).sum().item()

    # Print accuracy for each layer on the test set
    for i, cor in enumerate(correct):
        accuracy = cor / total
        print(f"Layer {i+1} Test Accuracy: {accuracy}")

    model.train()

    
# Layer 1 Test Accuracy: 0.1031
# Layer 2 Test Accuracy: 0.0534
# Layer 3 Test Accuracy: 0.151
# Layer 4 Test Accuracy: 0.9714
# Layer 5 Test Accuracy: 0.9714
