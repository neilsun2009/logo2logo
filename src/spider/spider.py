import numpy as np
from bs4 import BeautifulSoup
import requests

class spider():
  
  def __init__(self, url, list_name):
    self.url = url
    self.list_name = list_name
  
  def catch(self):
    wbdata = requests.get(self.url).text
    soup = BeautifulSoup(wbdata)
    return soup.select(self.list_name)