from dataset import CXRSegDataset
from model import SegmentationModel
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# load dataset as train/test split
dataset = CXRSegDataset()
train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# create segmentation model
model = SegmentationModel()
loss = torch.nn.BCEWithLogitsLoss()

# create optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# metric buffers
total_losses = []
total_acc = []

# run training loop
NUM_EPOCHS = 40
iteration = 0
for epoch in range(NUM_EPOCHS):
    # set model to train mode
    model.train()
    losses = []

    for img, label in train_loader:
        # log epoch
        print("Epoch:", epoch, "Iteration:", iteration, end='\r')
        iteration += 1

        # make mask prediction
        pred = model(img)

        # compute loss
        l = loss(pred, label)   
        losses.append(l.item())
        
        # compute gradiant
        optimizer.zero_grad()
        l.backward()

        # step optimizer
        optimizer.step()

    # analyze model at epoch
    model.eval()
    accs = []
    for img, label in test_loader:
        pred = model.get_mask_tensor(img)
        accs.append((pred == label).float().mean().item())

    # epoch complete
    print("Epoch:", epoch, "... Complete, Avg Loss:", np.mean(losses), "Avg Accuracy:", np.mean(accs))
    total_losses += losses
    total_acc += accs

    # step scheduler
    scheduler.step()

# save model
MODEL_NAME = 'model.th'
print("Model saved to", MODEL_NAME)
utils.save_model(model, MODEL_NAME)

# display loss/accuracy metrics
plt.title('Loss over Iterations')
plt.plot(range(len(total_losses)), total_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

plt.title('Validation Accuracy over Iterations')
plt.plot(range(len(total_acc)), total_acc)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()