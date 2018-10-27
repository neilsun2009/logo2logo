from sketch_generator.sketch import sketch_generator
from spider.json import spider_json

URL_CN = 'http://kotologo.com/json/d1926d56f827b804cacd46c22991e47a.json'
URL_DE = 'http://kotologo.com/json/deef1fc58a4df998a113c603dee700dc.json'
LIST_NAME = '#team_list li img'

# get images
images = spider_json(URL_DE).catch()

# filter and save
for i, image in enumerate(images):
  # print(i, image)
  sketch_generator(image, './dataset-raw/train/germany-%d-target.jpg' % i, './dataset-raw/train/germany-%d-input.jpg' % i).calculate()
  print('Completed %d/%d...' % (i, len(images)))

print('Finished! Congratulations!')