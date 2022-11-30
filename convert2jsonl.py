import json
import os
import argparse

parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
parser.add_argument(
    "-f",
    "--file",
    type=str,
    default=None,
)
args = parser.parse_args()

def toJson(lines):
    rlt = []
    for l in lines:
        l = l.strip()
        rlt.append(json.loads(l))
    return rlt

store_path = "./_data"
if not os.path.isdir(store_path):
    os.mkdir(store_path)


f = open(args.file, 'r')
lines = f.readlines()
js = toJson(lines)
json.dump(js, open(os.path.join(store_path, "input.json"), 'w'))



