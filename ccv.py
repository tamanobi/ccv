# -*- coding:utf-8 -*-
import sys
import numpy as np
import cv2

def QuantizeColor(img):
  rgb = cv2.split(img)
  for col in rgb:
    idx = np.where(col < 64)
    col[idx] = 32
    idx = np.where((64<=col)&(col<128))
    col[idx] = 96
    idx = np.where((128<=col)&(col<196))
    col[idx] = 160
    idx = np.where(196<=col)
    col[idx] = 224
  d_img = cv2.merge(rgb)
  return d_img

"""
Proccess of Computing CCV(color coherence vector)
@see http://vis.uky.edu/~cheung/courses/ee639_fall04/readings/ccv.pdf 
1. Blur
2. Quantizing color
3. Thresholding
4. Labeling
5. Counting
"""
def ccv(src, tau=0):
  img = src.copy()
  row, col, channels = img.shape
  if not col == 300:
    aspect = 300.0//col
    img = cv2.resize(img, None, fx=aspect, fy=aspect, interpolation = cv2.INTER_CUBIC)
  row, col, channels = img.shape
  # blur
  img = cv2.GaussianBlur(img, (3,3),0)
  # quantize color
  img = QuantizeColor(img)
  bgr = cv2.split(img)
  #bgr = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
  if tau == 0:
    tau = row*col * 0.1
  alpha = np.zeros(4)
  beta = np.zeros(4)
  # labeling
  for i,ch in enumerate(bgr):
    ret,th = cv2.threshold(ch,127,255,0)
    ret, labeled, stat, centroids = cv2.connectedComponentsWithStats(th, None, cv2.CC_STAT_AREA, None, connectivity=8)
    #!see https://github.com/atinfinity/lab/wiki/OpenCV%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%9F%E3%83%A9%E3%83%99%E3%83%AA%E3%83%B3%E3%82%B0#samplecode
    #!see http://docs.opencv.org/3.0.0/d3/dc0/group__imgproc__shape.html#gac7099124c0390051c6970a987e7dc5c5
    # generate ccv
    areas = [[v[4],label_idx] for label_idx,v in enumerate(stat)]
    coord = [[v[0],v[1]] for label_idx,v in enumerate(stat)]
    # Counting
    for a,c in zip(areas,coord):
      area_size = a[0]
      x,y = c[0], c[1]
      bin_idx = int(ch[y,x]//(64+1))
      if area_size >= tau:
        alpha[bin_idx] = alpha[bin_idx] + area_size
      else:
        beta[bin_idx] = beta[bin_idx] + area_size
  return alpha, beta

def ccv_plot(alpha, beta):
  import matplotlib.pyplot as plt
  X = [x for x in range(8)]
  Y = alpha.tolist()+beta.tolist()
  plt.bar(X, Y, align='center')
  plt.yscale('log')
  plt.xticks(X, (['alpha']*4)+(['beta']*4))
  plt.show()  

if __name__ == '__main__':
  argvs = sys.argv
  argc = len(argvs)
  img = cv2.imread(argvs[1])
  alpha, beta = ccv(img)
  print alpha.tolist()+beta.tolist()
  ccv_plot(alpha, beta)
