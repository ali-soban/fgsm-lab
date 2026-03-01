import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Define the Neural Network (Part 1, Point 1)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 2. Implementation of FGSM Attack Class (Part 1, Point 2)
class Attack:
    def __init__(self, model):
        self.model = model

    def fgsm_attack(self, image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image
        perturbed_image = image + epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

# 3. Training Function to ensure high Clean Accuracy
def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/60000] Loss: {loss.item():.6f}')

# 4. Evaluation Function (Part 1, Point 3)
def evaluate_robustness(model, device, test_loader, epsilon):
    model.eval()
    correct = 0
    adv_correct = 0
    total_samples = 500 # Using 500 samples for a solid statistical average
    processed = 0

    attacker = Attack(model)

    print(f"\nEvaluating with Epsilon: {epsilon}...")

    for data, target in test_loader:
        if processed >= total_samples:
            break
        
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        # Initial prediction
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        # If initial prediction is wrong, don't count for attack success
        if init_pred.item() != target.item():
            processed += 1
            continue

        correct += 1
        
        # Calculate loss and gradient
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        # Run FGSM
        perturbed_data = attacker.fgsm_attack(data, epsilon, data_grad)
        
        # Re-classify perturbed image
        output_adv = model(perturbed_data)
        final_pred = output_adv.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            adv_correct += 1
        
        processed += 1

    final_clean_acc = (correct / processed) * 100
    final_adv_acc = (adv_correct / processed) * 100

    print("-" * 40)
    print(f"Clean Accuracy: {final_clean_acc:.2f}%")
    print(f"Adversarial Accuracy: {final_adv_acc:.2f}%")
    print(f"Accuracy Drop: {final_clean_acc - final_adv_acc:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = 0.15 #higher for stronger attack 
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training model for 1 epoch to establish baseline...")
    train_model(model, device, train_loader, optimizer, 1)

    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model saved as mnist_model.pth")

    evaluate_robustness(model, device, test_loader, eps)