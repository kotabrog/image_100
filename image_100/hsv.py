import numpy as np

# Convert RGB to HSV(h: 0-360, s: 0-128, v:0-256)

def rgb_to_hsv(img):
	hsv = img.copy().astype('float32')
	rl = hsv[:, :, 0].copy()
	gl = hsv[:, :, 1].copy()
	bl = hsv[:, :, 2].copy()

	vmax = np.maximum(bl, np.maximum(gl, rl))
	vmin = np.minimum(bl, np.minimum(gl, rl))

	max_equal_min = (vmax == vmin)
	diff = np.maximum((vmax - vmin).astype('float32'), 1e-10)

	index = (max_equal_min == False) * (gl == vmin)
	hsv[:, :, 0][index] = 60 * (rl - bl)[index] / diff[index] + 300
	index = (max_equal_min == False) * (rl == vmin)
	hsv[:, :, 0][index] = 60 * (bl - gl)[index] / diff[index] + 180
	index = (max_equal_min == False) * (bl == vmin)
	hsv[:, :, 0][index] = 60 * (gl - rl)[index] / diff[index] + 60
	index = max_equal_min
	hsv[:, :, 0][index] = 0

	hsv[:, :, 1] = vmax - vmin
	hsv[:, :, 2] = vmax
	return hsv

# Convert RGB to HSV(h: 0-360, s: 0-128, v:0-256)

def hsv_to_rgb(hsv):
	img = hsv.copy()
	h = img[:, :, 0].copy()
	s = img[:, :, 1].copy()
	v = img[:, :, 2].copy()

	h2 = h / 60
	x = (s * (1 - np.abs(h2 % 2 - 1)))

	img[:, :, 0] = v - s
	img[:, :, 1] = v - s
	img[:, :, 2] = v - s

	index = (h2 < 1)
	img[:, :, 0][index] += s[index]
	img[:, :, 1][index] += x[index]
	index = (1 <= h2) * (h2 < 2)
	img[:, :, 0][index] += x[index]
	img[:, :, 1][index] += s[index]
	index = (2 <= h2) * (h2 < 3)
	img[:, :, 1][index] += s[index]
	img[:, :, 2][index] += x[index]
	index = (3 <= h2) * (h2 < 4)
	img[:, :, 1][index] += x[index]
	img[:, :, 2][index] += s[index]
	index = (4 <= h2) * (h2 < 5)
	img[:, :, 0][index] += x[index]
	img[:, :, 2][index] += s[index]
	index = 5 <= h2
	img[:, :, 0][index] += s[index]
	img[:, :, 2][index] += x[index]

	return np.clip(np.round(img), 0, 255).astype('uint8')

# reversing hue

def turn_hsv(img):
	hsv = rgb_to_hsv(img)
	hsv[:, :, 0] = (hsv[:, :, 0] + 180) % 360
	return hsv_to_rgb(hsv)