import cv2
import argparse
def load_image():
parser = argparse.ArgumentParser()
parser.add_argument("-i","--image",required = True, help = "Path to the image")
args = vars(parser.parse_args())
image = cv2.imread(args["image"])
cv2.imshow("image",image)
cv2.waitKey(0)
def filter_obj(image):
  min_red = np.array([0,100,80])
  max_red = np.array([10,256,256])
def 

def find_obj(image):
  image = load_image()
  image = cv2.cvtColor(image,cv2.BGR2RGB)
  max_dimension = max(image.shape)
  scale = 700/max_dimension #shape image
  # image stride 
  image = cv2.resize(image, None, fx= scale,fy= scale)
  
  image_blur = cv2.GaussianBlur(image, (7,7),0)
  image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
  
  
