import numpy as np
import warnings

# Convert RGB to grayscale

def rgb_to_gray(img):
	b = img[:, :, 0].copy()
	g = img[:, :, 1].copy()
	r = img[:, :, 2].copy()
	gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
	return gray.astype('uint8')

# Convert grayscale to binary image

def gray_to_2gray(img, thresh=128):
	return np.minimum(img // thresh, 1) * 255

# Convert RGB to binary image

def rgb_to_2gray(img, thresh=128):
	return gray_to_2gray(rgb_to_gray(img), thresh)

# Convert grayscale to binary image Otsu's binarization and return thresh and binary image

def otsu_thresh(img):
	thresh_list = np.array([i for i in range(256)])
	count_class0 = np.array([np.sum(img < thresh) for thresh in thresh_list])
	w0 = np.array([np.sum(img < thresh) for thresh in thresh_list])
	w1 = img.size - count_class0
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)
		m0 = np.array([img[img < thresh].mean() for thresh in thresh_list])
		m1 = np.array([img[img >= thresh].mean() for thresh in thresh_list])
	dist_bet_classes = (m0 - m1) ** 2 * w0 * w1 / (w0 + w1) / (w0 + w1)
	dist_bet_classes = np.nan_to_num(dist_bet_classes)
	max_th = np.argmax(dist_bet_classes)
	binary_img = gray_to_2gray(img, max_th)
	return max_th, binary_img

# Convert RGB to binary image Otsu's binarization and return thresh and binary image

def rgb_to_otsu_thresh(img):
	gray = rgb_to_gray(img)
	return otsu_thresh(gray)