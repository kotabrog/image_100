import matplotlib.pyplot as plt

# show two images

def show_img_origin_and_conversion(img1, img2, cmap = 'viridis', cmap_orig = 'viridis'):
	plt.figure(figsize=(12, 3))
	plt.subplot(1, 2, 1)
	plt.title('input')
	plt.imshow(img1, cmap=cmap_orig)
	plt.subplot(1, 2, 2)
	plt.title('answer')
	plt.imshow(img2, cmap=cmap)
	plt.show()

# show original image and h, s, v

def show_img_origin_and_hsv(img_orig, img_hsv):
	plt.figure(figsize=(12, 3))
	plt.subplot(1, 4, 1)
	plt.title('input')
	plt.imshow(img_orig)

	plt.subplot(1, 4, 2)
	plt.title('Hue')
	plt.imshow(img_hsv[..., 0] / 360, cmap='hsv')

	plt.subplot(1, 4, 3)
	plt.title('Saturation')
	plt.imshow(img_hsv[..., 1] / 128, cmap='gray')

	plt.subplot(1, 4, 4)
	plt.title('Value')
	plt.imshow(img_hsv[..., 2], cmap='gray')

	plt.show()