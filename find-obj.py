from __future__ import division
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from math import cos,sin
green = (0,255,0)
def show(image):
  plt.figure(figsize = (10,10))
  plt.imshow(image, interpolation = 'nearest')

def overlay_mask(mask, image):
  rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
  img = cv2.addWeighted(rgb_mask,0.5, image,0.5,0)
  return img
def find_biggest_coutour(image):
  image = image.copy()
  coutour, hierarchy = cv2.findCoutours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
  
  coutour_size = [(cv2.contourArea(contour),contour)for contour in coutours]
  biggest_contour = (contour_sizes, key = lambda x: x[0])[1]
  #tra ve gia tri co duong vien mau do lon nhat
  mask = np.zeros(image.shape, np.uint8)
  cv2.drawContours(mask,[biggest_contour],-1, 255,-1)
  return mask, biggest_contour
def circle_contour(image,contour):
  # khoanh vung duong vien cua hinh anh can tim lai
  image_with_ellipse = image.copy()
  ellipse = cv2.fitEllipse(contour)
  # cong chung lai tao thanh hinh can tim
  cv2.ellipse(image_with_ellipse,ellipse,green,2,cv2.CV_AA)
  return image_with_ellipse
def load_image():
  parse = argparse.ArgumentParser()
  parse.add_argument("-i","--image",required = True, help = "Path to the image")
  args = vars(parser.parse_args())
  image = cv2.imread(args["image"])
  cv2.imshow("image",image)
  cv2.waitKey(0)
def filter_obj(image):
  min_red = np.array([0,100,80])
  max_red = np.array([10,256,256])

def find_obj(image):
  # buoc 1 : tim va phan loai 
  image = load_image()
  image = cv2.cvtColor(image,cv2.BGR2RGB)
  max_dimension = max(image.shape)
  scale = 700/max_dimension #shape image
  # image stride 
  image = cv2.resize(image, None, fx= scale,fy= scale)
  # lam tron cac thanh phan cua matran
  image_blur = cv2.GaussianBlur(image, (7,7),0)
  image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
  # buoc 2 khoi tao filter 
  min_red = np.array([0,100,80])
  max_red = np.array([10,256,256])
  
  # khoi tao cac chieu cua ma tran lan luot cua hinh anh Red
  mask1 = cv2.inRange(image_blur,min_red,max_red)
  #khoi tao filter diem sang
  min_red2 = np.array([170,100,80])
  max_red2 = np.array([180,256,256])
  mask2 = cv2.inrange(image_blur,min_red2,max_red2)
  
  # ket hop 2 mask 
  mask = mask1 + mask2
  
  # buoc 3 tim vat the
  # phan doan hinh anh phong doan vat the can tim
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
  mask_closed = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)
  mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
  
  # buoc 4 du doan tim hinh anh mong muon 
  # tim vi tri mau xac xuat cao nhat + khoanh vung vien cua hinh anh
  big_strawberry_contour, mask_strawberries = find_biggest_countour(mask_clean)
  
  # tim gioi han mau trong image
  overplay = overplay_mask(mask_clean, image)
  
  # khoanh vung hinh anh
  # bang cach tim gioi han mau va so sanh voi hinh anh can tim
  circled = circle_contour(overplay, big_strawberry_contour)
  show(circled)
  # buoc 5 chuyen doi hinh anh sang hinh goc cua image
  bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
  return bgr
# doc hinh anh can tin
image = cv2.imread()
result = find_obj(image)
# viet no thanh hinh anh kac
cv2.imwrite('',result)
