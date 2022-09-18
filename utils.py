# imports 
from torch import autocast
from diffusers import StableDiffusionPipeline
import torch
import cv2
import numpy as np
from torch import nn
from PIL import Image
from numpy import asarray

# model call
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
model.eval()
lenet = nn.Sequential(*list(model.children())[:-2])

# function
def score(prompt):
      ## generate image and it's feature vector
  with autocast("cuda"):
    image = pipe(prompt, guidance_scale=10)["sample"][0]  
  s_image = image
  image = cv2.resize(np.asarray(image), (224, 224))
  image = image.transpose([2, 0, 1])
  features = lenet(torch.unsqueeze(torch.Tensor(image), axis=0)).squeeze().detach().numpy()
  s_image.save("./static/image.png")

  ### read the constant image
  img = Image.open('./static/eyeImage.jpg')
  numpydata = asarray(img)
  image2 = cv2.resize(np.asarray(numpydata), (224, 224))
  image2 = image2.transpose([2, 0, 1])
  features2 = lenet(torch.unsqueeze(torch.Tensor(image2), axis=0)).squeeze().detach().numpy()

  # generate score
  A = features
  B = features2

  
  # from sklearn.metrics import mean_squared_error
  # return mean_squared_error(A,B)
  
  from numpy.linalg import norm
  ans = (np.dot(A,B)/(norm(A)*norm(B)))
  print(ans)
  return int((ans**2.5)*100)
