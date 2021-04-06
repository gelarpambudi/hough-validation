import os

txt_file = open('image.txt', 'w')
for file in os.listdir('image'):
    txt_file.write(os.path.join('image/',file)+'\n')