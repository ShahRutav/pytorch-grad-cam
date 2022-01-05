import cv2
import numpy as np
from pytorch_grad_cam import GradCAM, GradCAMRL
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import resnet34
from PIL import Image
import gym
import mj_envs

env = gym.make('kitchen_knob2_off-v3')
rgb_img = env.sim.render(width=224, height=224, camera_name='left_cap')
rgb_img = rgb_img[::-1,:,:]
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])


model = resnet34(pretrained=True)
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
target_category = None

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
print(grayscale_cam.shape)
print(rgb_img.shape)
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
visualization_pil = Image.fromarray(visualization)
visualization_pil.save("test.png")

