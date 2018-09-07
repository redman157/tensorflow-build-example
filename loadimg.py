import cv2
import argparse
def load_image():
  parser = argparse.ArgumentParser()
  parser.add_argument("-i","--image",required = True, help = "Path to the image")
  args = vars(parser.parse_args())
  image = cv2.imread(args["image"])
  cv2.imshow("image",image)
  cv2.waitKey(0)

