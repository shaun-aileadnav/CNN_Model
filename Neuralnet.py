import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import wandb

x, y = torch.load(r"C:\Users\shaun\Documents\Projects\CNN_Model\data\training.pt")

y_new = F.one_hot(y, num_classes=10)

class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x / 255.
        self.y = F.one_hot(self.y, num_classes=10).to(float)
    def __len__(self): 
        return self.x.shape[0]
    def __getitem__(self, ix): 
        return self.x[ix], self.y[ix]

train_ds = CTDataset(r"C:\Users\shaun\Documents\Projects\CNN_Model\data\training.pt")
test_ds = CTDataset(r"C:\Users\shaun\Documents\Projects\CNN_Model\data\test.pt")

xs, ys = train_ds[0:4]

# Define the split sizes
train_size = int(0.8 * len(train_ds))  # 80% for training
val_size = len(train_ds) - train_size  # 20% for validation

# Split the dataset
train_subset, val_subset = torch.utils.data.random_split(train_ds, [train_size, val_size])

# Create DataLoaders for both training and validation datasets
train_dl = DataLoader(train_subset, batch_size=5)
val_dl = DataLoader(val_subset, batch_size=5)

L = nn.CrossEntropyLoss()

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2,100)
        self.Matrix2 = nn.Linear(100,50)
        self.Matrix3 = nn.Linear(50,10)
        self.R = nn.ReLU()
    def forward(self,x):
        x = x.view(-1,28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()

f = MyNeuralNet()

model_file_path = "models/my_model.pth"

def train_model(dl, val_dl, f, n_epochs=40):
    # Optimization
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    # Train model
    losses = []
    val_losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(dl)
        epoch_loss = 0  # Initialize epoch loss
        for i, (x, y) in enumerate(dl):
            # Update the weights of the network
            opt.zero_grad() 
            loss_value = L(f(x), y) 
            loss_value.backward() 
            opt.step() 
            # Store training data
            # losses.append(loss_value.item())
            # Accumulate training loss for the epoch
            epoch_loss += loss_value.item()

        # Average the loss for the epoch
        losses.append(epoch_loss / N)

        # Validation step
        with torch.no_grad():
            val_loss = 0
            for x_val, y_val in val_dl:
                val_loss += L(f(x_val), y_val).item()
            val_losses.append(val_loss / len(val_dl))
            epochs.append(epoch)  # Store epoch number for each validation loss
            print(f'Validation Loss: {val_loss / len(val_dl)}')

    torch.save(f.state_dict(), model_file_path)
    return np.array(epochs), np.array(losses), np.array(val_losses)

def new():
    run = wandb.init(project="MNIST_FASTAPI_APP")

    epoch_data, loss_data, val_loss_data = train_model(train_dl, val_dl, f)

    plt.plot(epoch_data, loss_data, label='Training Loss')
    plt.plot(epoch_data, val_loss_data, label='Validation Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    model_artifact = wandb.Artifact('my_neural_net', type='model')
    model_artifact.add_file(model_file_path)
    wandb.log_artifact(model_artifact)
    run.link_artifact(model_artifact, "ai-leadnav-org/wandb-registry-model/<Shauns_models>")
    wandb.finish

    xs, ys = train_ds[0:2000]
    yhats = f(xs).argmax(axis=1)

    fig, ax = plt.subplots(10,4,figsize=(10,15))
    for i in range(40):
        plt.subplot(10,4,i+1)
        plt.imshow(xs[i])
        plt.title(f'Predicted Digit: {yhats[i]}')
    fig.tight_layout()
    plt.show()

    xs, ys = test_ds[:2000]
    yhats = f(xs).argmax(axis=1)

    fig, ax = plt.subplots(10,4,figsize=(10,15))
    for i in range(40):
        plt.subplot(10,4,i+1)
        plt.imshow(xs[i])
        plt.title(f'Predicted Digit: {yhats[i]}')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    new()
