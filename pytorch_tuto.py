#import libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets 
from torchvision.transforms import ToTensor ,Lambda
import matplotlib.pyplot as plt

#Downloading the dataset from open datasets
train_dataset=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),

)
test_dataset=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64 

# Create data loaders.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader)) #Iteration with 64 samples
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
#Print the last element 
img = train_features[63].squeeze() # from [1,28,28] to [28,28]
label = train_labels[63]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

#Building the model
class ClothesClass(nn.Module):
  def __init__(self):
    super(ClothesClass,self).__init__()
    self.flatten=nn.Flatten()
    self.my_network=nn.Sequential(
        nn.Linear(28*28,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10),
        nn.ReLU()
    )
  def forward(self,img):
    img=self.flatten(img)
    Model=self.my_network(img)
    return Model

#Creating instance
model = ClothesClass().to(device) 
print(model)

#Move to GPU (GPU is faster than cpu)
device='cuda' if torch.cuda.is_available() else 'cpu'

#Optimizing the model parameters


#Hyperparameters 
learning_rate = 1e-3 
batch_size = 64 #Number of samples in each iteration
epochs = 5 #Iteration number

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #Stochastic gradient descent

epochs = 40 # If the epochs increase the accurary increase 
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

#Saving the model 

torch.save(model.state_dict(), "data/my_model.pth")

print("Saved PyTorch Model State to my_model.pth")

#Loading the model

model = ClothesClass()
model.load_state_dict(torch.load("data/my_model.pth"))

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
x, y = train_dataset[64][0], train_dataset[64][1]
with torch.no_grad():
    pred = model(x)
    print(pred)
    predicted = classes[pred[0].argmax(0)]
    print(f'Predicted: "{predicted}"')