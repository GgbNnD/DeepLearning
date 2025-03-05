from torchvision import transforms
from PIL import Image

image_path ="F:\\DeepLearning\\Dataset\\train\\ants_image\\0013035.jpg"
img = Image.open(image_path)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)