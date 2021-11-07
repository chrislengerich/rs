import torch
import cv2

# image picker which outputs the image as a Tensor to |img_tensor|
from PIL import Image
import numpy as np

def load_image(image_path, x32=False):
  img = cv2.imread(image_path).astype(np.float32)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  h, w = img.shape[:2]

  if x32: # resize image to multiple of 32s
      def to_32s(x):
          return 256 if x < 256 else x - x%32
      img = cv2.resize(img, (to_32s(w), to_32s(h)))

  img = torch.from_numpy(img)
  img = img/127.5 - 1.0
  return img

# image chooser dialogue
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
img_path = filedialog.askopenfilename()
image = load_image(img_path)

model = torch.hub.load('bryandlee/animegan2-pytorch', 'generator').eval()

device = torch.device('cpu')
with torch.no_grad():
  input = image.permute(2, 0, 1).unsqueeze(0).to(device)
  out = model(input).squeeze(0).permute(1, 2, 0).cpu().numpy()
  out = (out + 1) * 127.5
  out = np.clip(out, 0, 255).astype(np.uint8)

path = img_path.split('.')
outpath = path[0] + "_anime.jpg"

cv2.imwrite(outpath, cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
print(f"image saved")



