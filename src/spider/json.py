# get logo from kotologo.com using json

import numpy as np
import math
import requests
import json

class spider_json():

  BASE_URL = 'http://img.kotologo.com/150px_logo/'
  
  def __init__(self, url, is_club=True):
    self.url = url
    self.is_club = is_club

  def get_image_url(self, id):
    if str.isdigit(id): # some teams maybe reserve teams, therefore suffixxed with alphabet
      id = int(id)
      return '%s%d/%d.gif' % (self.BASE_URL, math.floor(id/10000), id)
    else:
      return False
  
  def catch(self):
    rawdata = requests.get(self.url).text
    jsondata = json.loads(rawdata)
    
    # print(jsondata)
    images = []
    # logos for clubs
    if (self.is_club):
      jsondata = jsondata['ll']
      # logo in ch
      if 'ch' in jsondata:
        for ch in jsondata['ch']:
          if 'tl' in ch:
            for tl in ch['tl']:
              if 'id' in tl:
                for iidd in tl['id']:
                  url = self.get_image_url(iidd)
                  if url: images.append(url)
            # for one level of league with partitions
            # it is likely to be a low level league
            # therefore we only need one layer of that most of the times
            break 
                  
      # logo in tl
      if 'tl' in jsondata:
        for tl in jsondata['tl']:
          if 'id' in tl:
            for iidd in tl['id']:
              url = self.get_image_url(iidd)
              if url: images.append(url)
    # logos for national teams
    else:
      jsondata = jsondata['rl']
      if 'tl' in jsondata:
        tl = jsondata['tl'][0]
        if 'id' in tl:
          for iidd in tl['id']:
            url = self.get_image_url(iidd)
            if url: images.append(url)
    # for img in image_list:
    #   images.append(img.get('src'))
    #   print(img)
    return images
