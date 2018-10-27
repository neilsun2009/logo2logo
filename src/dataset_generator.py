from sketch_generator.sketch import sketch_generator
from spider.json import spider_json
import json
import os

CLUB_CONFIG_FILE = './data_config/club.json'
NATION_CONFIG_FILE = './data_config/nation.json'
LIST_NAME = '#team_list li img'

# get dataset config
with open(CLUB_CONFIG_FILE) as f:
  club_config = json.loads(f.read())
  club_config = club_config['data']
with open(NATION_CONFIG_FILE) as f:
  nation_config = json.loads(f.read())
  nation_config = nation_config['data']

current_id = 5
# clubs
# get images
# images = spider_json(club_config[current_id]['url']).catch()

# print('Getting data from %s...' % club_config[current_id]['country'])

# # check for directory
# filepath = '../dataset-raw/' + club_config[current_id]['country']
# if not os.path.exists(filepath):
#   os.makedirs(filepath)

# # filter and save
# for i, image in enumerate(images):
#   # print(i, image)
#   sketch_generator(image, '%s/%s-%d-target.jpg' 
#     % (filepath, club_config[current_id]['country'], i), 
#     '%s/%s-%d-input.jpg' 
#     % (filepath, club_config[current_id]['country'], i)).calculate()
#   print('Completed %d/%d...' % (i, len(images)))

# national teams
images = spider_json(nation_config[current_id]['url'], is_club=False).catch()

print('Getting data from %s...' % nation_config[current_id]['continent'])

# check for directory
filepath = '../dataset-raw/' + nation_config[current_id]['continent']
if not os.path.exists(filepath):
  os.makedirs(filepath)

# filter and save
for i, image in enumerate(images):
  # print(i, image)
  sketch_generator(image, '%s/%s-%d-target.jpg' 
    % (filepath, nation_config[current_id]['continent'], i), 
    '%s/%s-%d-input.jpg' 
    % (filepath, nation_config[current_id]['continent'], i)).calculate()
  print('Completed %d/%d...' % (i, len(images)))

print('Finished! Congratulations!')