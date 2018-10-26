import cv2
import numpy as np
import math
import imageio

class sketch_generator():

  # file: path for input image file
  # *_output: two output path for output images
  # size: the height/weight for output image size
  def __init__(self, file, resize_output, sketch_output, size=256):
    self.file = file
    self.resize_output = resize_output
    self.sketch_output = sketch_output
    self.size = size

  # mocking a jittering effect on an image
  # it uses a small window to slide along the image
  # within each window, the image will move both horizontally and vertically (-max_transition, max_transition)
  def _jittering(self, img, window_width, window_height, max_transition=1):
    height, width = img.shape
    new_img = np.zeros((height, width))
    h_times = math.ceil(height / window_height)
    w_times = math.ceil(width / window_width)
    for i in range(h_times):
      for j in range(w_times):
        # (-max_transition, max_transition)
        transition = np.random.randint(-max_transition, max_transition)
        # window start and end
        wh_start = max(0, i*window_height + transition)
        wh_end = min(height, (i+1)*window_height + transition)
        ww_start = max(0, j*window_width + transition)
        ww_end = min(width, (j+1)*window_width + transition)
        # output start and end
        oh_start = i * window_height
        oh_end = oh_start + wh_end - wh_start
        ow_start = j * window_width
        ow_end = ow_start + ww_end - ww_start
        new_img[oh_start:oh_end, ow_start:ow_end] = img[wh_start:wh_end, ww_start:ww_end]
    return new_img.astype(np.uint8)

  # fade out the initial picture randomly on each pixel
  # within the range of (max_fading, 1)
  # i.e. the result pixel will be pixel * random(max_fading, 1)
  def _random_fading(self, img, max_fading=0.7):
    mask = np.random.rand(*img.shape) * (1-max_fading) + max_fading
    return (img.astype(np.float32) * mask).astype(np.uint8)

  # calculate resize and sketch
  def calculate(self):
    # use imageio to read in gif
    img = imageio.imread(self.file)
    # img = cv2.imread('./test/suning-small.gif', cv2.IMREAD_GRAYSCALE)

    # get cv2-formatted BGR and transfer transparent to white 
    img_color = img[:,:,(2, 1, 0)]
    if img.shape[2] > 3:
      img_color[img[:,:,3]==0] = (255, 255, 255)

    # resize to 300*300
    img = cv2.resize(img_color, (self.size, self.size), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(self.resize_output, img)

    # convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussian denoise
    img = cv2.GaussianBlur(img,(5,5),0)

    # get a black bg with white line edged pic
    output = cv2.Canny(img, 50, 20)

    # apply jittering
    output = self._jittering(output, 15, 15)
    # random fading
    output = self._random_fading(output)
    # bilateral smoothing
    output = cv2.bilateralFilter(output, 50, 75, 75)
    # inverse
    output = 255 - output

    cv2.imwrite(self.sketch_output, output)