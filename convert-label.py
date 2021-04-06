import json
import os
import glob


json_label = os.listdir("gt-json/")
for file in json_label:
    f_json = open(os.path.join("gt-json/",file), 'r')
    data = json.loads(f_json.read())
    label_txt = file.rsplit(".", 1)[0] + '.txt'
    print(label_txt)
    label_txt = os.path.join("gt-txt/", label_txt)
    f_txt = open(label_txt, "w")

    for i in data['shapes']:
        points = i['points']
        f_txt.write(str(points[0][0])+', '+str(points[0][1])+', '+str(points[1][0])+', '+str(points[1][1])+', ')
        f_txt.write(str(data['imageHeight'])+', '+str(data['imageWidth'])+'\n')
    
    f_txt.close()