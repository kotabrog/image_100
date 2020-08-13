# Convert RGB to GBR

def rgb_to_gbr(img):
	return img[:, :, ::-1]