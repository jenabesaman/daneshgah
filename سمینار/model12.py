import time

from torch.optim.lr_scheduler import StepLR
import torch
import cv2
import numpy
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
start=time.time()


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])

        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale

        if self.transform:

            image = self.transform(Image.fromarray(image))

        return image



# transform = transforms.Compose([
#     transforms.Resize((224, 320)),
#     transforms.Grayscale(num_output_channels=3),
#     # transforms.RandomHorizontalFlip(),
#     # transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

transform = transforms.Compose([transforms.Resize((224, 320)),
                                        transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# img_path = ("C:/Workarea/File_Analyser/etc/not_melli/download.jpg")
# image = Image.open("C:/Workarea/File_Analyser/etc/not_melli/download.jpg")
# image=CustomDataset("C:/Workarea/File_Analyser/etc/not_melli/download.jpg")


# image = cv2.imread(img_path)  # Read image in grayscale
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# _, thresholded_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
# image = transform(thresholded_image)
# plt.imshow(thresholded_image,cmap='gray')
# plt.axis(False)
# plt.show()

# image = transform(img_path)
# image=image.squeeze(0).permute(1,2,0)
# plt.imshow(image)
# plt.axis(False)
# plt.show()


data_dir = 'C:/works/File_Analyser/dataset/melli'
dataset = CustomDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)
model = model.to(device)
# model.load_state_dict(torch.load("C:/works/File_Analyser/Models/model11.pth", map_location='cpu'))

#
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 33
for epoch in range(num_epochs):
    for inputs in dataloader:
        inputs = inputs.to(device)
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.ones_like(outputs))  # Target label is 1 (positive)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f}')

torch.save(obj=model.state_dict(), f="C:/works/File_Analyser/Models/model12.pth")
#

#
model.load_state_dict(torch.load("C:/Workarea/File_Analyser/main/models/model11.pth", map_location=device))
# def predict_image(image_path):
#     image = Image.open(image_path)
#     image = transform(image).unsqueeze(0)
#     image = image.to(device)
#
#     with torch.no_grad():
#         output = model(image)
#
#     prob = torch.sigmoid(output)
#
#     return prob.item()


# def predict_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = transform(Image.fromarray(image)).unsqueeze(0)
#     # _, image = cv2.threshold(image, 138, 255, cv2.THRESH_BINARY)
#     # image = transform(image).unsqueeze(0)
#     image = image.to(device)
#
#     with torch.no_grad():
#         output = model(image)
#
#     prob = torch.sigmoid(output)
#
#     return prob.item()

def predict_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = transform(Image.fromarray(image)).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    prob = torch.sigmoid(output)
    return prob.item()


new_image_path = 'C:/works/pytorch/pytorch_learning/File_Analyser/etc/not_melli/1234.jpg'
prediction = predict_image(new_image_path)
print(prediction)

if prediction > 0.94:
    print("The image belongs to the class.")
else:
    print("The image does not belong to the class.")

end=time.time()
print(f"timing {end-start}")