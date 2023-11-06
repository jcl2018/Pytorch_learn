
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

### 1. Create nn.Dataset ###
# Here we just download
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

### 2. Create Dataloader ###
# This wraps an iterable over our dataset, and supports automatic batching, sampling, shuffling and multiprocess data loading


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Test...
# dataset is iterable
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    print("The first y: ", y[0])
    break


### 3. Create the model class ###
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module): # Must be an instance of nn.Module
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # Linear layer #1 (N, 28*28) -> (N, 512)
            nn.ReLU(),
            nn.Linear(512, 512), # Linear layer #2 (N, 512) -> (N, 512)
            nn.ReLU(),
            nn.Linear(512, 10) # Linear layer #3 (N, 512) -> (N, 10)
        )

    def forward(self, x): # Must implement this function to define forward pass.
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) # The "logit" before Softmax activation.
        return logits

model = NeuralNetwork().to(device)
print(model)


### 4. Define loss function and optimizer ###
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


### 5. Define training loop ###
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # The total number of data point / batch size
    model.train()  # Change model to "training" state. For example, save gradients.
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # X, y are tensors. By default, they are CPU tensors. to(device) can move them to gpu

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward() # Start backpropagation: starting from loss
        optimizer.step() # optimizer does one step (one minibatch)
        optimizer.zero_grad() # zeroing-out gradients for this mini-batch, so it won't affect the next mini-batch

        # report loss and iteration step
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Minibatch loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

### 6. Define test loop ###
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() # .item converts tensor to numpy array.
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # how many data points are predicted correctly
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg minibatch loss: {test_loss:>8f} \n")

### 7. main function ###

# Main function for train and test
# epochs = 5 # one epoch visits all data points.
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")
#
# # Save model parameters
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

### 8. Main function to load model and predict ###
model = NeuralNetwork().to(device) # Note here we reuse the model class, but just load parameters
model.load_state_dict(torch.load("model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')