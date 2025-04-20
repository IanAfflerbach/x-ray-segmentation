from dataset import CXRSegDataset
from model import SegmentationModel
import utils
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# load dataset as train/test split
dataset = CXRSegDataset()
train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# create segmentation model
model = SegmentationModel()
loss = torch.nn.BCEWithLogitsLoss()

# create optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# run training loop
NUM_EPOCHS = 20
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

    # epoch complete
    print("Epoch:", epoch, "... Complete, Avg Loss:", sum(losses) / len(losses))

    # step scheduler
    scheduler.step()

# visual performance of model
test_iter = iter(test_loader)
imgs, masks = [], []
for _ in range(5):
    img_tensor, mask_tensor = next(test_iter)
    imgs.append(transforms.ToPILImage()(img_tensor[0]))
    masks.append(transforms.ToPILImage()(mask_tensor[0]))
stitched_img = utils.stitch_images_horizontal(*imgs)
stitched_masks = utils.stitch_images_horizontal(*masks)

true_segs = [utils.segment_image(imgs[i], masks[i]) for i in range(5)]
stitched_true_segs = utils.stitch_images_horizontal(*true_segs)

pred_masks = [model.get_mask(img) for img in imgs]
stitched_pred_masks = utils.stitch_images_horizontal(*pred_masks)

pred_segs = [utils.segment_image(imgs[i], pred_masks[i]) for i in range(5)]
stitched_pred_segs = utils.stitch_images_horizontal(*pred_segs)

stitched_all = utils.stitch_images_vertical(stitched_img, stitched_masks, stitched_true_segs, stitched_pred_masks, stitched_pred_segs)
stitched_all.show()