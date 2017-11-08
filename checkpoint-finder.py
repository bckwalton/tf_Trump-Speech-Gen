import re
import os
list_of_files = os.listdir()

def extract_number(f):
    s = re.findall("(\d+).data-00000-of-00001", f)
    return (int(s[0]) if s else -1,f)

target = (max(list_of_files,key=extract_number))
target_name = target.split('.')
