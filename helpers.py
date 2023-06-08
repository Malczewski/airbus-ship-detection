import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

TRAIN_PATH = './data/train_v2/'

def get_image_path(image_id):
	return TRAIN_PATH + image_id

def decode_pixels(rle_encoding: str, shape: tuple):
	mask = np.zeros(shape, dtype=int)
	for start, length in rle_encoding:
		end = start + length
		mask.flat[start:end] = 1
	
	return np.rot90(np.fliplr(mask))

def str2rle(rle_string: str):
	if isinstance(rle_string, str) and len(rle_string.strip()) > 0:
		str_pairs = re.findall(r"\d+ \d+", rle_string)
		return list(map(lambda str_pair: list(map(lambda str_num: int(str_num), str_pair.split(' '))), str_pairs))
	else: 
		return []
	
def get_mask(rle_string: str, shape: tuple):
	return decode_pixels(str2rle(rle_string), shape)

def render_pair(title, image, mask):
	plt.imshow(image)
	plt.imshow(mask, alpha=0.3)
	plt.axis('off')
	plt.title(title)
	plt.show()

def get_image(img_name):
	image_path = get_image_path(img_name)
	image = Image.open(image_path)
	return np.array(image)

def get_image_mask(rle: str, shape: tuple):
	return get_mask(rle, shape)

def get_bbox(mask):
	# Find the non-zero indices
	nonzero_indices = np.nonzero(mask)

	# Calculate the bounding box coordinates
	min_row = np.min(nonzero_indices[0])
	max_row = np.max(nonzero_indices[0])
	min_col = np.min(nonzero_indices[1])
	max_col = np.max(nonzero_indices[1])
	return (min_row, min_col), (max_row, max_col)

def get_image_region(image, center_x, center_y, shape):
	width = shape[1]
	height = shape[0]
	top_left_x = max(center_x - shape[1] // 2, 0)
	top_left_y = max(center_y - shape[0] // 2, 0)

	if top_left_x + width > image.shape[1]:
		top_left_x = image.shape[1] - width
	if top_left_y + height > image.shape[0]:
		top_left_y = image.shape[0] - height

	# Define slice for rows and columns
	rows_slice = slice(top_left_y, top_left_y + height)
	cols_slice = slice(top_left_x, top_left_x + width)
	# Extract the square region from the image
	return image[rows_slice, cols_slice]

def crop_image(row, shape):
	image = get_image(row[0])
	mask = get_image_mask(row[1], image.shape[:2])
	top_left, bottom_right = get_bbox(mask)
	center_x = (top_left[1] + bottom_right[1]) // 2
	center_y = (top_left[0] + bottom_right[0]) // 2
	resized_image = get_image_region(image, center_x, center_y, shape)
	resized_mask = get_image_region(mask, center_x, center_y, shape)
	return resized_image, resized_mask

def cut_image(image, square_size):
	width, height = image.size
	x_steps = int(np.ceil(width / square_size))
	y_steps = int(np.ceil(height / square_size))

	squares = []
	
	for y in range(y_steps):
		for x in range(x_steps):
			x_start = x * square_size
			y_start = y * square_size
			x_end = min(x_start + square_size, width)
			y_end = min(y_start + square_size, height)
			
			square = image.crop((x_start, y_start, x_end, y_end))
			
			if x_end - x_start < square_size or y_end - y_start < square_size:
				# Extend the square to full size by creating a new image
				extended_square = Image.new('RGB', (square_size, square_size), (0, 0, 0))
				extended_square.paste(square, (0, 0))
				squares.append(np.array(extended_square))
			else:
				squares.append(np.array(square))
	
	return np.array(squares)

def join_images(image_arrays, square_size, image_shape):
	x_steps = int(np.ceil(image_shape[1] / square_size))
	y_steps = int(np.ceil(image_shape[0] / square_size))
	image = np.zeros((image_shape[0], image_shape[1], 1))
	for y in range(y_steps):
		for x in range(x_steps):
			x_start = x * square_size
			y_start = y * square_size
			image[y_start:y_start + square_size, x_start:x_start + square_size] = image_arrays[y * x_steps + x]
	return image