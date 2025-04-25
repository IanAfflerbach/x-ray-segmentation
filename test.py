from dataset import CXRSegDataset
from model import SegmentationModel
import utils
from torch.utils.data import DataLoader
from torchvision import transforms

# save model
MODEL_NAME = 'model.th'
print("Load model", MODEL_NAME)
model = utils.load_model(SegmentationModel, MODEL_NAME)

# load dataset as train/test split
dataset = CXRSegDataset()
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# visual performance of model
print("Creating example predictions")
test_iter = iter(data_loader)
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