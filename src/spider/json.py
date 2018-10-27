# get logo from kotologo.com using json

import numpy as np
import math
import requests
import json

class spider_json():

  BASE_URL = 'http://img.kotologo.com/150px_logo/'
  
  def __init__(self, url):
    self.url = url

  def get_image_url(self, id):
    return '%s%d/%d.gif' % (self.BASE_URL, math.floor(id/10000), id)
  
  def catch(self):
    rawdata = requests.get(self.url).text
    jsondata = json.loads(rawdata)
    jsondata = jsondata['ll']
    # print(jsondata)
    images = []
    # logo in ch
    if 'ch' in jsondata:
      for ch in jsondata['ch']:
        if 'tl' in ch:
          for tl in ch['tl']:
            if 'id' in tl:
              for iidd in tl['id']:
                images.append(self.get_image_url(int(iidd)))
    # logo in tl
    if 'tl' in jsondata:
      for tl in jsondata['tl']:
        if 'id' in tl:
          for iidd in tl['id']:
            images.append(self.get_image_url(int(iidd)))
    # for img in image_list:
    #   images.append(img.get('src'))
    #   print(img)
    return images

  def __call__(self):
    return self.catch()