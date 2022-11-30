import json
import argparse
from tw_rouge import get_rouge
import os

def main(args):
    refs, preds = {}, {}

    with open(args.reference) as file:
        for line in file:
            line = json.loads(line)
            refs[line['id']] = line['title'].strip() + '\n'

    with open(args.submission) as file:
        for line in file:
            try:
                line = json.loads(line)
            except:
                print(line)
                exit()
            preds[line['id']] = line['title'].strip() + '\n'

    keys =  refs.keys()
    refs = [refs[key] for key in keys]
    preds = [preds[key] for key in keys]

    rlt = json.dumps(get_rouge(preds, refs), indent=2)
    print(rlt)
    with open(os.path.join(args.store_dir, "score.json"), 'w') as f:
        f.write(rlt)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference')
    parser.add_argument('-s', '--submission')
    parser.add_argument('-d', '--store_dir')
    args = parser.parse_args()
    main(args)
