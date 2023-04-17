import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

from CAM.myGradCam import myGradCam
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load model
pthfile = '\...'
model = torch.load(pthfile)
model = model.eval().to(device)

# choose visualization layer
target_layers = model.pre_net.layer2
second_image_path = '\...'
second_img_pil = Image.open(second_image_path)
second_img_pil = second_img_pil.convert('RGB')
second_img_pil = second_img_pil.resize((512, 512))

# visualization images
image_path = 'xxx.jpg'
first_image_path = image_path
first_image_pil = Image.open(first_image_path)
first_image_pil = first_image_pil.convert('RGB')
first_image_pil = first_image_pil.resize((512, 512))
test_transform = transforms.Compose([
                                     # transforms.RandomCrop(112),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

first_input_tensor = test_transform(first_image_pil)
first_input_tensor = first_input_tensor.unsqueeze(0).to(device)

second_input_tensor = test_transform(second_img_pil)
second_input_tensor = second_input_tensor.unsqueeze(0).to(device)

input_tensor = [first_input_tensor, second_input_tensor]

# class type
targets = [ClassifierOutputTarget(2)]

cam = myGradCam(model=model, target_layers=target_layers, use_cuda=True)
if hasattr(torch.cuda, 'empty_cache'):
	torch.cuda.empty_cache()
cam_map = cam(input_tensor=input_tensor, targets=targets)[0]

cam_map[cam_map > 0.1] += 0.4

plt.imshow(cam_map)
plt.show()

result = overlay_mask(first_image_pil, Image.fromarray(cam_map), alpha=0.6)
# save images
result.show()
result.save(image_path)
