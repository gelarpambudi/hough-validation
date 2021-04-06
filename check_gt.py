from glob import glob
import os
import traceback

def get_gt():
    line = []
    for image_file in glob("gt-txt/*.txt"):
        with open(image_file, 'r') as file:
            gt = file.readlines()
        line.extend(gt)
    return len(line)

if __name__ == "__main__":
    try:
        result = get_gt()
        print("Number of Lines: ", result)
    except Exception as e:
        traceback.print_exc()