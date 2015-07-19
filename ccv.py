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

#!see http://homepage3.nifty.com/ishidate/vcpp_color/vcpp_color.htm
def ColorPalette(n=64):
  pallete = np.empty((1,n,3),dtype=np.uint8)
  pallete.fill(255)
  hsv = cv2.cvtColor(pallete, cv2.COLOR_BGR2HSV)
  hsv_ch = cv2.split(hsv)
  hsv_ch[0].fill(255)
  hsv_ch[1].fill(255)
  for i in range(n):
    hsv_ch[0][0,i] = i*255/n
    if i % 2 == 1:
      hsv_ch[2][0,i] = 128
  color_palette = cv2.cvtColor(cv2.merge(hsv_ch), cv2.COLOR_HSV2BGR)
  colors = [[pix for pix in color_palette[0,i]] for i in range(n)]
  #cv2.imshow('palette', cv2.resize(color_palette, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST))
  #cv2.waitKey(0)
  return colors

def ccv(src):
  img = src.copy()
  row, col, channels = img.shape
  if not row == 300:
    aspect = 300.0/row
    img = cv2.resize(img, None, fx=aspect, fy=aspect, interpolation = cv2.INTER_CUBIC)
  row, col, channels = img.shape
  # blur
  img = cv2.GaussianBlur(img, (3,3),0)
  # quantize color
  img = QuantizeColor(img)
  bgr = cv2.split(img)
  #bgr = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
#colors = ColorPalette(64)
  tau = row*col * 0.1
  alpha = np.zeros(4)
  beta = np.zeros(4)
  # labeling
  for i,ch in enumerate(bgr):
    ret,th = cv2.threshold(ch,127,255,0)
    #image, cnts, hierarchy = cv2.findContours(th,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    ret, labeled, stat, centroids = cv2.connectedComponentsWithStats(th, None, cv2.CC_STAT_AREA, None, connectivity=8)
#    label_img = np.empty(img.shape, dtype=np.uint8)
#    for y in range(row):
#      for x in range(col):
#        idx = labeled[y,x]
#        if idx > -1:
#          label_img[y,x] = colors[idx]
#        else:
#          label_img[y,x] = 0
    #!see https://github.com/atinfinity/lab/wiki/OpenCV%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%9F%E3%83%A9%E3%83%99%E3%83%AA%E3%83%B3%E3%82%B0#samplecode
    #!see http://docs.opencv.org/3.0.0/d3/dc0/group__imgproc__shape.html#gac7099124c0390051c6970a987e7dc5c5
    # generate ccv
    areas = [[v[4],label_idx] for label_idx,v in enumerate(stat)]
    coord = [[v[0],v[1]] for label_idx,v in enumerate(stat)]
    for a,c in zip(areas,coord):
      x,y = c[0], c[1]
      idx = ch[y,x]/(64+1)
      if a[0] >= tau:
        alpha[idx] = alpha[idx] + a[0]
      else:
        beta[idx] = beta[idx] + a[0]
  return alpha, beta

if __name__ == '__main__':
  argvs = sys.argv
  argc = len(argvs)
  img = cv2.imread('./query/test20.jpg')
  alpha, beta = ccv(img)
  print np.linalg.norm(alpha-beta)
  print alpha+beta
