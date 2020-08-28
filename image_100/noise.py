import numpy as np

# Salt-and-pepper noise

def salt_and_pepper_noise(img, num):
	img_noise = img.copy()

	img_row = np.random.randint(0, img_noise.shape[0] - 1 , num)
	img_col = np.random.randint(0, img_noise.shape[1] - 1 , num)
	img_noise[(img_row, img_col)] = (255, 255, 255)

	img_row = np.random.randint(0, img_noise.shape[0] - 1 , num)
	img_col = np.random.randint(0, img_noise.shape[1] - 1 , num)
	img_noise[(img_row, img_col)] = (0, 0, 0)
	return img_noise

