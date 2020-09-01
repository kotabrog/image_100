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

# Gaussian filter

def gaussian_filter(img, kernel=(3, 3), sigma=1.3):

	def two_dimensional_gaussian(index=(0, 0), sigma=1.3):
		x = index[0]
		y = index[1]
		return 1 / (2 * np.pi * sigma * sigma) *\
				np.exp(-(index[0] ** 2 + index[1] ** 2) / (2 * sigma ** 2))

	half_kernel_h = kernel[0] // 2
	half_kernel_w = kernel[1] // 2
	half_under_h = (kernel[0] - 1) // 2
	half_right_w = (kernel[1] - 1) // 2
	half_adjust_h = 0 if kernel[0] % 2 else 0.5
	half_adjust_w = 0 if kernel[1] % 2 else 0.5
	gaussian_filter =\
		[[two_dimensional_gaussian(
			(i - half_kernel_h + half_adjust_h, 
			j - half_kernel_w + half_adjust_w))
		for j in range(kernel[1])]
		for i in range(kernel[0])]
	gaussian_filter = gaussian_filter / np.sum(gaussian_filter)

	pad_img = np.pad(img, [(half_kernel_h, half_under_h), 
							(half_kernel_w, half_right_w), (0, 0)], 'reflect') 
	pad_img = np.array([[[np.sum(gaussian_filter * 
							pad_img[i: i + kernel[0], j: j + kernel[1], k])
							for k in range(3)]
							for j in range(img.shape[1])]
							for i in range(img.shape[0])])
	return np.round(pad_img).astype('uint8')

# Median filter

def median_filter(img, kernel=(3, 3)):
	half_kernel_h = kernel[0] // 2
	half_kernel_w = kernel[1] // 2
	half_under_h = (kernel[0] - 1) // 2
	half_right_w = (kernel[1] - 1) // 2

	pad_img = np.pad(img, [(half_kernel_h, half_under_h), 
							(half_kernel_w, half_right_w), (0, 0)], 'symmetric') 
	pad_img = np.array([[[np.median(pad_img[i: i + kernel[0], 
											j: j + kernel[1], k])
							for k in range(3)]
							for j in range(img.shape[1])]
							for i in range(img.shape[0])])
	return np.round(pad_img).astype('uint8')

# Smooth filter

def mean_filter(img, kernel=(3, 3)):
	half_kernel_h = kernel[0] // 2
	half_kernel_w = kernel[1] // 2
	half_under_h = (kernel[0] - 1) // 2
	half_right_w = (kernel[1] - 1) // 2

	pad_img = np.pad(img, [(half_kernel_h, half_under_h), 
							(half_kernel_w, half_right_w), (0, 0)], 'reflect') 
	pad_img = np.array([[[np.mean(pad_img[i: i + kernel[0], 
										j: j + kernel[1], k])
							for k in range(3)]
							for j in range(img.shape[1])]
							for i in range(img.shape[0])])
	return np.round(pad_img).astype('uint8')
