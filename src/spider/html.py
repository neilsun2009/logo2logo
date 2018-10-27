import numpy as np
from bs4 import BeautifulSoup
import requests

class spider_html():
  
  def __init__(self, url, list_name):
    self.url = url
    self.list_name = list_name
  
  def catch(self):
    wbdata = requests.get(self.url).text
    soup = BeautifulSoup(wbdata, "html.parser")
    print(wbdata)
    image_list = soup.select(self.list_name)
    print(image_list)
    images = []
    for img in image_list:
      images.append(img.get('src'))
      print(img)
    return images

  def __call__(self):
    return self.catch()