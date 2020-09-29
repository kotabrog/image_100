import numpy as np
from .gray import rgb_to_gray

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

# Motion filter

def motion_filter(img, kernel=(3, 3)):
	half_kernel_h = kernel[0] // 2
	half_kernel_w = kernel[1] // 2
	half_under_h = (kernel[0] - 1) // 2
	half_right_w = (kernel[1] - 1) // 2
	size = min(kernel[0], kernel[1])
	pad_img = np.pad(img, [(half_kernel_h, half_under_h), 
							(half_kernel_w, half_right_w), (0, 0)], 'reflect') 
	pad_img = np.array([[[np.mean([pad_img[i + l, j + l, k] for l in range(size)])
							for k in range(3)]
							for j in range(img.shape[1])]
							for i in range(img.shape[0])])
	return np.round(pad_img).astype('uint8')

# Max-min filter

def max_min_filter(img, kernel=(3, 3)):
	if len(img.shape) == 3:
		_img = rgb_to_gray(img)
	else:
		_img = img
	half_kernel_h = kernel[0] // 2
	half_kernel_w = kernel[1] // 2
	half_under_h = (kernel[0] - 1) // 2
	half_right_w = (kernel[1] - 1) // 2
	pad_img = np.pad(_img, [(half_kernel_h, half_under_h), 
							(half_kernel_w, half_right_w)], 'edge') 
	pad_img = np.array([[np.max(pad_img[i: i + kernel[0], 
										j: j + kernel[1]])
							- np.min(pad_img[i: i + kernel[0], 
										j: j + kernel[1]])
							for j in range(img.shape[1])]
							for i in range(img.shape[0])])
	return np.round(pad_img).astype('uint8')

# Differential filter

def differential_filter(img, mode='y'):
	if len(img.shape) == 3:
		_img = rgb_to_gray(img)
	else:
		_img = img
	if mode == 'y':
		dif_img = np.pad(_img, [(1, 0), (0, 0)], 'reflect').astype(int)
		return np.abs(dif_img[:-1, :] - _img).astype('uint8')
	else:
		dif_img = np.pad(_img, [(0, 0), (1, 0)], 'reflect').astype(int)
		return np.abs(dif_img[:, :-1] - _img).astype('uint8')

# Prewitt filter

def prewitt_filter(img, k_size=(3, 3), mode='y'):
	if len(img.shape) == 3:
		_img = rgb_to_gray(img)
	else:
		_img = img
	p_filter = np.zeros(k_size)
	if mode == 'y':
		if k_size[0] == 1:
			return None
		p_filter[0] = 1
		p_filter[-1] = -1
	else:
		if k_size[1] == 1:
			return None
		p_filter[:, 0] = 1
		p_filter[:, -1] = -1
	half_k_size_h = k_size[0] // 2
	half_k_size_w = k_size[1] // 2
	half_under_h = (k_size[0] - 1) // 2
	half_right_w = (k_size[1] - 1) // 2
	pad_img = np.pad(_img, [(half_k_size_h, half_under_h), 
							(half_k_size_w, half_right_w)], 'reflect')
	pad_img = np.array([[np.abs(np.mean(p_filter * 
							pad_img[i: i + k_size[0], j: j + k_size[1]]))
							for j in range(img.shape[1])]
							for i in range(img.shape[0])])
	return np.round(pad_img).astype('uint8')

# sobel filter

def sobel_filter(img, k_size=(3, 3), mode='y'):
	if len(img.shape) == 3:
		_img = rgb_to_gray(img)
	else:
		_img = img
	half_k_size_h = k_size[0] // 2
	half_k_size_w = k_size[1] // 2
	half_under_h = (k_size[0] - 1) // 2
	half_right_w = (k_size[1] - 1) // 2
	s_filter = np.zeros(k_size)
	if mode == 'y':
		if k_size[0] == 1:
			return None
		s_filter[0] = 1
		s_filter[-1] = -1
		s_filter[0, half_k_size_w] = 2
		s_filter[-1, half_k_size_w] = -2
		if k_size[0] % 2 == 0:
			s_filter[0, half_k_size_w - 1] = 2
			s_filter[-1, half_k_size_w - 1] = -2
	else:
		if k_size[1] == 1:
			return None
		s_filter[:, 0] = 1
		s_filter[:, -1] = -1
		s_filter[half_k_size_h, 0] = 2
		s_filter[half_k_size_h, 1] = -2
		if k_size[0] % 2 == 0:
			s_filter[half_k_size_h - 1, 0] = 2
			s_filter[half_k_size_h - 1, -1] = -2
	pad_img = np.pad(_img, [(half_k_size_h, half_under_h), 
							(half_k_size_w, half_right_w)], 'reflect')
	pad_img = np.array([[np.sum(s_filter * 
							pad_img[i: i + k_size[0], j: j + k_size[1]])
							for j in range(img.shape[1])]
							for i in range(img.shape[0])])
	pad_img -= pad_img.min()
	pad_img /= pad_img.max()
	return pad_img

# laplacian filter

def laplacian_filter(img):
	if len(img.shape) == 3:
		_img = rgb_to_gray(img)
	else:
		_img = img
	l_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
	pad_img = np.pad(_img, (1, 1), 'reflect')
	pad_img = np.array([[np.sum(l_filter * 
							pad_img[i: i + 3, j: j + 3])
							for j in range(img.shape[1])]
							for i in range(img.shape[0])])
	pad_img -= pad_img.min()
	pad_img /= pad_img.max()
	return pad_img

# emboss filter

def emboss_filter(img):
	if len(img.shape) == 3:
		_img = rgb_to_gray(img)
	else:
		_img = img
	e_filter = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float)
	pad_img = np.pad(_img, (1, 1), 'reflect')
	pad_img = np.array([[np.sum(e_filter * 
							pad_img[i: i + 3, j: j + 3])
							for j in range(img.shape[1])]
							for i in range(img.shape[0])])
	pad_img -= pad_img.min()
	pad_img /= pad_img.max()
	return pad_img
