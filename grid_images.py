# This is a helper file
# which will combine multiple images into one image
# in a user-defined grid

import os
import sys
import numpy as np
from PIL import ImageGrab, Image

print("Welcome to Grid Images Script")
print("You can enter 'quit' to exit")

# get user-defined grid shape
grid_shape = None
while grid_shape is None:
    shape = input("Enter grid shape in (num_horizontal, num_vertical) format: ")
    if shape == "quit": sys.exit(0)
    shape = shape.strip("()").split(",")
    try:
        if len(shape) != 2:
            raise ValueError
        shape_x = int(shape[0])
        shape_y = int(shape[1])
    except Exception as e:
        print("Error: {}".format(e))
        continue
    else:
        grid_shape = (shape_x, shape_y)

images_count = grid_shape[0] * grid_shape[1]
image_list = []
# get images from clipboard
image_idx = 0
while image_idx < images_count:
    output = input("Copy an image and press ENTER ({} remaining): ".format(images_count - image_idx - 1))
    if output == "quit": sys.exit(0)
    new_image = ImageGrab.grabclipboard()
    if new_image is None:
        print("Cannot find copyed image. Please try again")
        continue
    if isinstance(new_image, list):
        print("Cannot process multiple copyed images. Please copy one by one")
        continue
    new_image_converted = Image.new("RGB", new_image.size, (255, 255, 255))
    new_image_converted.paste(new_image)
    image_list.append(new_image_converted)
    image_idx += 1

# compute grid size
grid_size = np.array([x.size for x in image_list])
grid_size = np.max(grid_size, axis=0).tolist()
grid_size_final = [x*y for (x,y) in zip(grid_size, grid_shape)]

grid = Image.new('RGB', grid_size_final)
for i in range(grid_shape[1]): # vertical
    for j in range(grid_shape[0]): # horizontal
        location = (j*grid_size[0], i*grid_size[1])
        grid.paste(image_list[j + i*grid_shape[0]], location)

filename = input("Enter filename with extension (example.png): ")
if filename == "quit": sys.exit(0)
filename = "example.png" if filename == "" else filename

while os.path.exists(filename):
    result = input("File already exists. Overwrite? (Y/N): ").lower()
    if result == "quit" or result == "n": sys.exit(0)
    elif result == "y":
        os.remove(filename)
        break
    else:
        print("Please enter 'Y' or 'N'")

grid.save(filename, quality=100)