"""
ymliu@2024.1.23
convert fcgec.test to predict.json for online evaluation
"""

import os
import json
from zipfile import ZipFile
import argparse
from collections import OrderedDict


def main(args):
    submit = json.load(open(args.input), object_pairs_hook=OrderedDict)
    pred = open(args.pred).read().strip().split("\n\n")
    outdir = os.path.dirname(args.out)
    assert len(submit) == len(pred)
    for uid, para in zip(submit, pred):
        src = para.split("\n")[0].split("\t")[1]
        tgt = para.split("\n")[1].split("\t")[1]
        assert submit[uid]['sentence'] == src
        submit[uid]['error_flag'] = 0 if src == tgt else 1
        submit[uid]['error_type'] = "*"
        submit[uid]['correction'] = tgt
    tmp_json = os.path.join(outdir, "predict.json")
    json.dump(submit, open(tmp_json, "w"), ensure_ascii=False, indent=4)

    with ZipFile(args.out, 'w') as f_zip:
        f_zip.write(tmp_json, "predict.json")
    if not args.keep_json:
        os.remove(tmp_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/FCGEC_test.json", help="offical FCGEC_test.json path", required=True)
    parser.add_argument("--pred", help="predict fcgec test path", required=True)
    parser.add_argument("--out", help="submit zip path", required=True)
    parser.add_argument("--keep_json", action='store_true')
    args = parser.parse_args()
    main(args)
