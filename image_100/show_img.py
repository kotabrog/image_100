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