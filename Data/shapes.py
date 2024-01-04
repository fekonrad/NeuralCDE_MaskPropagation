import numpy as np
from numpy import random
import cv2
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RandomShapes(Dataset):
    def __init__(self, num_images, height=512, width=512):
        self.num_images = num_images
        self.img, self.mask = generate_shapes(num_images, height, width)
        # Define a transformation to normalize the images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # Assuming RGB images
        ])

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = self.img[idx]
        image = self.transform(image)
        mask = torch.tensor(self.mask[idx])
        return image.to(torch.float32), mask.to(torch.float32)


def generate_shapes(num, height=512, width=512, background=False):
    """
    generates 'num' images of shape (height,widht) with randomly placed shapes (triangles, rectangles, circles)
    returns images and corresponding binary masks for circular shapes
    """
    x = np.ndarray((num, height, width, 3))
    y = np.ndarray((num, height, width, 1))
    for i in range(num):
        img = np.zeros((height, width, 3))
        mask = np.zeros((height, width, 1))
        num_shapes = random.randint(0, 5)   # generates between 2 to 8 shapes per image
        for _ in range(num_shapes):
            shape_id = random.randint(0, 3)
            if shape_id == 0:   # rectangle
                x1 = random.randint(0, width - 1)
                y1 = random.randint(0, height - 1)
                x2 = random.randint(0, width - 1)
                y2 = random.randint(0, height - 1)

                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            if shape_id == 1:   # triangle
                vertices = np.array([[random.randint(0, width-1), random.randint(0, height-1)],
                                     [random.randint(0, width-1), random.randint(0, height-1)],
                                     [random.randint(0, width-1), random.randint(0, height-1)]])
                vertices = vertices.reshape((-1, 1, 2))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.fillPoly(img, [vertices], color)

            if shape_id == 2:   # circle
                center = (random.randint(0, width-1), random.randint(0, height-1))
                radius = random.randint(int(0.2*height), int(0.4*height))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.circle(img, center, radius, color, -1)
                cv2.circle(mask, center, radius, (1, 1, 1), -1)  # binary mask corresponding to the circle

        img_rgb = cv2.cvtColor(cv2.convertScaleAbs(img), cv2.COLOR_BGR2RGB)
        x[i] = img_rgb
        y[i] = mask
    return x, y.transpose(0, 3, 1, 2)


class MovingShapes(Dataset):
    def __init__(self, num_images, height=128, width=128, frames=20):
        self.num_images = num_images
        self.frames = frames
        self.height = height
        self.width = width
        self.init_shapes = self.init_shapes(num_images=num_images, height=height, width=width)
        # Define a transformation to normalize the images
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def init_shapes(self, num_images, height, width):
        images = []
        for k in range(num_images):
            types = ["triangle", "circle", "rectangle"]
            shapes = []
            for i in range(5):  # generates 5 shapes
                shape = types[random.randint(0, 3)]
                if shape == "triangle":
                    vertices = np.array([[random.randint(0, width - 1), random.randint(0, height - 1)],
                                         [random.randint(0, width - 1), random.randint(0, height - 1)],
                                         [random.randint(0, width - 1), random.randint(0, height - 1)]])
                    vertices = vertices.reshape((-1, 1, 2))
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    shapes.append(["triangle", vertices, color])

                if shape == "circle":
                    center = (random.randint(0, width - 1), random.randint(0, height - 1))
                    radius = random.randint(int(0.1 * height), int(0.2 * height))
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    shapes.append(["circle", [center, radius], color])

                if shape == "rectangle":
                    x1 = random.randint(0, width - 1)
                    y1 = random.randint(0, height - 1)
                    x2 = random.randint(0, width - 1)
                    y2 = random.randint(0, height - 1)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    shapes.append(["rectangle", [(x1, y1), (x2, y2)], color])
            images.append(shapes)
        return images

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        shapes = self.init_shapes[idx]
        image0, mask0 = self.generate_image(shapes, height=self.height, width=self.width)
        images = np.empty((self.frames, self.height, self.width, 3), dtype=np.uint8)
        images[0:1] = image0
        masks = np.zeros((self.frames, self.height, self.width, 1))
        masks[0] = mask0

        for i in range(1, self.frames):
            for k in range(5):
                step = np.around(np.random.normal(0.0, 1, 2)).astype(int)
                if shapes[k][0] == "triangle":
                    shapes[k][1][0] += step  # update vertices
                    shapes[k][1][1] += step
                    shapes[k][1][2] += step

                if shapes[k][0] == "circle":
                    shapes[k][1][0] += step  # update center

                if shapes[k][0] == "rectangle":
                    shapes[k][1][0] += step  # update vertices
                    shapes[k][1][1] += step

            img0, mask0 = self.generate_image(shapes, height=self.height, width=self.width)
            images[i] = img0
            masks[i] = mask0

        images_tensor = torch.from_numpy(images.transpose(0, 3, 1, 2)).float() / 255.0
        masks_tensor = torch.from_numpy(masks.transpose(0, 3, 1, 2)).float()
        return images_tensor, masks_tensor

    def generate_image(self, shapes, height, width):
        """
        :param shapes: list, as generated in __init__shapes, describing the position and color of various shapes
        :return: an image of shape (height,width,3) containing the shapes
        """
        img = np.zeros((height, width, 3))
        mask = np.zeros((height, width, 1))
        for shape in shapes:
            if shape[0] == "triangle":
                cv2.fillPoly(img, [shape[1]], shape[2])

            if shape[0] == "circle":
                cv2.circle(img, shape[1][0], shape[1][1], shape[2], -1)
                cv2.circle(mask, shape[1][0], shape[1][1], (1, 1, 1), -1)  # binary mask corresponding to the circle

            if shape[0] == "rectangle":
                cv2.rectangle(img, shape[1][0], shape[1][1], shape[2], -1)

        img_rgb = cv2.cvtColor(cv2.convertScaleAbs(img), cv2.COLOR_BGR2RGB)
        return img_rgb, mask


