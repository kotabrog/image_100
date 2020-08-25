import numpy as np

# reduce colors

def color_subtraction(img, num_division=4):
	div_value = 256 // num_division
	return np.clip(img // div_value * div_value + div_value // 2, 0, 255)

# avarage pooling

def avarage_pooling(img, block_size=(8, 8)):
	img_h, img_w, img_c= img.shape
	mean_img_h = (img_h - 1) // block_size[0] + 1
	mean_img_w = (img_w - 1) // block_size[1] + 1
	return_img = np.empty((mean_img_h * block_size[0],
			mean_img_w * block_size[1], img_c), dtype='uint8')
	for i in range(0, img_h, block_size[0]):
		for j in range(0, img_w, block_size[1]):
			for k in range(img_c):
				return_img[i: i + block_size[0],
				j: j + block_size[1], k] = np.mean(
				img[i: min(i + block_size[0], img_h),
				j: min(j + block_size[1], img_w), k])
	return return_img[:img_h, :img_w, :img_c]

# max pooling

def max_pooling(img, block_size=(8, 8)):
	img_h, img_w, img_c= img.shape
	mean_img_h = (img_h - 1) // block_size[0] + 1
	mean_img_w = (img_w - 1) // block_size[1] + 1
	return_img = np.empty((mean_img_h * block_size[0],
			mean_img_w * block_size[1], img_c), dtype='uint8')
	for i in range(0, img_h, block_size[0]):
		for j in range(0, img_w, block_size[1]):
			for k in range(img_c):
				return_img[i: i + block_size[0],
				j: j + block_size[1], k] = np.max(
				img[i: min(i + block_size[0], img_h),
				j: min(j + block_size[1], img_w), k])
	return return_img[:img_h, :img_w, :img_c]