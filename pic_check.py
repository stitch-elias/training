import cv2
import numpy as np


class BlurDetector(object):

    def __init__(self):
        """Initialize a DCT based blur detector"""
        self.dct_threshold = 8.0
        self.max_hist = 0.1
        self.hist_weight = np.array([8, 7, 6, 5, 4, 3, 2, 1,
                                     7, 8, 7, 6, 5, 4, 3, 2,
                                     6, 7, 8, 7, 6, 5, 4, 3,
                                     5, 6, 7, 8, 7, 6, 5, 4,
                                     4, 5, 6, 7, 8, 7, 6, 5,
                                     3, 4, 5, 6, 7, 8, 7, 6,
                                     2, 3, 4, 5, 6, 7, 8, 7,
                                     1, 2, 3, 4, 5, 6, 7, 8
                                     ]).reshape(8, 8)
        self.weight_total = 344.0

    def check_image_size(self, image, block_size=8):
        """Make sure the image size is valid.
        Args:
            image: input image as a numpy array.
            block_size: the size of the minimal DCT block.
        Returns:
            result: boolean value indicating whether the image is valid.
            image: a modified valid image.
        """
        result = True
        height, width = image.shape[:2]
        _y = height % block_size
        _x = width % block_size

        pad_x = pad_y = 0

        if _y != 0:
            pad_y = block_size - _y
            result = False
        if _x != 0:
            pad_x = block_size - _x
            result = False

        image = cv2.copyMakeBorder(
            image, 0, pad_y, 0, pad_x, cv2.BORDER_REPLICATE)

        return result, image

    def get_blurness(self, image, block_size=8):
        """Estimate the blurness of an image.
        Args:
            image: image as a numpy array of shape [height, width, channels].
            block_size: the size of the minimal DCT block size.
        Returns:
            a float value represents the blurness.
        """
        # A 2D histogram.
        hist = np.zeros((8, 8), dtype=int)

        # Only the illumination is considered in blur.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Split the image into patches and do DCT on the image patch.
        height, width = image.shape
        round_v = int(height / block_size)
        round_h = int(width / block_size)
        for v in range(round_v):
            for h in range(round_h):
                v_start = v * block_size
                v_end = v_start + block_size
                h_start = h * block_size
                h_end = h_start + block_size

                image_patch = image[v_start:v_end, h_start:h_end]
                image_patch = np.float32(image_patch)

                patch_spectrum = cv2.dct(image_patch)

                patch_none_zero = np.abs(patch_spectrum) > self.dct_threshold
                hist += patch_none_zero.astype(int)

        _blur = hist < self.max_hist * hist[0, 0]
        _blur = (np.multiply(_blur.astype(int), self.hist_weight)).sum()
        return _blur/self.weight_total


def LaplaceFilter(img):
    h, w, c = img.shape

    K_size = 3

    # 零填充
    pad = K_size // 2
    out = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=np.float64)
    out[pad:pad + h, pad:pad + w] = img.copy().astype(np.float64)

    K = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad + y, pad + x, ci] = np.sum(K * tmp[y:y + K_size, x:x + K_size, ci])

    out = out[pad:pad + h, pad:pad + w].astype(np.uint8)
    return out

def detect_blur_fft(gray,size,thresh):
    cy,cx = gray.shape[0]//2,gray.shape[1]//2
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    fft_shift[cy-size:cy+size,cx-size:cx+size]=0
    fft_shift=np.fft.ifftshift(fft_shift)
    recon = np.fft.ifft2(fft_shift)
    spectrum = 20*np.log(np.abs(recon))
    mean = np.mean(spectrum)
    blurry = mean <=thresh
    return mean,blurry

import math

def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res

def custom_blur_demo(image):
    # kernels = np.ones([5, 5], np.float32)/25
    kernels = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst = cv2.filter2D(image, -1, kernel=kernels)
    return dst
