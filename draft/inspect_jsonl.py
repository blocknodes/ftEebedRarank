import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as file:
    first_line = file.readline()
    first_record = json.loads(first_line)
    print(first_record)