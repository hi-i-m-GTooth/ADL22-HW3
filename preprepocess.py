import json
import os

def toJson(lines):
    rlt = []
    for l in lines:
        l = l.strip()
        rlt.append(json.loads(l))
    return rlt

names = ["public", "sample_submission", "sample_test", "train"]
store_path = "./_data"
if not os.path.isdir(store_path):
    os.mkdir(store_path)


for n in names:
    f = open(os.path.join("./data", n+".jsonl"), 'r')
    lines = f.readlines()
    js = toJson(lines)
    json.dump(js, open(os.path.join(store_path, n+".json"), 'w'))



