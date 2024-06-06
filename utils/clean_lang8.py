"""
ymliu@2023.12.19
data.train url: http://tcci.ccf.org.cn/conference/2018/dldoc/trainingdata02.tar.gz
filter_sentences.txt url: https://github.com/HillZhang1999/MuCGEC/blob/main/data/MuCGEC/filter_sentences.txt
"""

import re
import emoji
import opencc
from tqdm import tqdm
from collections import OrderedDict

with open("data.train") as f_in, open('filter_sentences.txt') as mucgec_filter_sentences:
    with open('filter_sentences_lang8.txt') as mucgec_filter_sentences_lang8, open("lang8.train", "w") as f_out:
        converter = opencc.OpenCC('t2s.json')  # Convert Chinese Traditional to Simplified Chinese
        filter_num = 0
        filter_sentences = set()
        filter_sentences_lang8 = set()
        has_filtered_sentemces = set()
        for line in mucgec_filter_sentences:  # filter out data that overlaps with MuCGEC
            line = line.strip().replace("\u3000", "")
            if line == "到我忘带雨伞妹妹像一个大人批评了我，然后舍给了我她的小黄伞。":
                filter_sentences.add("看到我忘带雨伞妹妹像一个大人批评了我，然后舍给了我她的小黄伞。")
            else:
                filter_sentences.add(line)

        for line in mucgec_filter_sentences_lang8:  # filter out data that overlaps with MuCGEC
            line = line.strip()
            if line == "到我忘带雨伞妹妹像一个大人批评了我，然后舍给了我她的小黄伞。":
                filter_sentences_lang8.add("看到我忘带雨伞妹妹像一个大人批评了我，然后舍给了我她的小黄伞。")
            else:
                filter_sentences_lang8.add(line)

        converter = opencc.OpenCC('t2s.json')  # convert chinese traditional to simplified chinese
        count_source_sentence = 0
        count_target_sentence = 0
        data = OrderedDict()

        for line in tqdm(f_in, desc="Clean Lang-8"):
            line = line.strip().split("\t")
            source_sentence = line[2].strip().replace(" ", "").replace("\u200b", "").replace("\u3000", "")
            source_sentence = emoji.replace_emoji(source_sentence, '')
            target_sentences = line[3:]
            if source_sentence in filter_sentences:
                filter_num += 1
                has_filtered_sentemces.add(source_sentence)
                continue
            if len(target_sentences) != int(line[1]):  # mostly noisy sentences
                continue
            if len(target_sentences) == 0:
                continue
            if not re.search('[^=,.?!@#$%^&\*()_+:"<>/\[\]\\`~–—，。、《》？;；‘’：“”【】、{}|·！￥…（）」●＊-]', source_sentence):
                continue  # purely punctuation
            if source_sentence not in data.keys():
                data[source_sentence] = []
            for target_sentence in target_sentences:  # clean data
                if len(source_sentence) > 1.5 * len(target_sentence):
                    continue
                if len(target_sentence) > 1.5 * len(source_sentence):
                    continue
                target_sentence = target_sentence.strip().replace(" ", "").replace("\u200b", "").replace("\u3000", "")  # remove whitespace
                if target_sentence == source_sentence:
                    continue
                target_sentence = re.sub("&gt;", ">", target_sentence)
                target_sentence = re.sub("&lt;", "<", target_sentence)
                target_sentence = target_sentence.lstrip('•●＊\*-–—”,，\.。、\|=：>·･#<_+\]\}】;；’？」')  # remove noisy punctuation
                target_sentence = target_sentence.rstrip('#{')  # remove noisy punctuation
                target_sentence = target_sentence.strip()  # remove whitespace
                target_sentence = re.sub("。{4,}", "……", target_sentence)
                target_sentence = re.sub("。{1,2,3}", "。", target_sentence)
                target_sentence = re.sub("\.{2,}", "……", target_sentence)
                target_sentence = re.sub("？+", "？", target_sentence)
                target_sentence = re.sub("\?+", "?", target_sentence)
                target_sentence = re.sub("，+", "，", target_sentence)
                target_sentence = re.sub(",+", ",", target_sentence)
                target_sentence = re.sub("；+", "；", target_sentence)
                target_sentence = re.sub(";+", ";", target_sentence)
                target_sentence = re.sub("：+", "：", target_sentence)
                target_sentence = re.sub("、+", "、", target_sentence)
                target_sentence = re.sub("-{2,}", "——", target_sentence)
                target_sentence = re.sub("–+", "——", target_sentence)
                target_sentence = re.sub("—+", "——", target_sentence)
                target_sentence = re.sub("⋯+", "……", target_sentence)
                target_sentence = re.sub("·{2,}", "……", target_sentence)
                target_sentence = re.sub("･{2,}", "……", target_sentence)
                target_sentence = re.sub("…+？+", "？", target_sentence)
                target_sentence = re.sub("…+\?+", "?", target_sentence)
                target_sentence = re.sub("？+…+", "？", target_sentence)
                target_sentence = re.sub("\?+…+", "?", target_sentence)
                target_sentence = re.sub("…+。+", "……", target_sentence)
                target_sentence = re.sub("。+…+", "……", target_sentence)
                target_sentence = re.sub("…+\.+", "……", target_sentence)
                target_sentence = re.sub("\.+…+", "……", target_sentence)
                target_sentence = re.sub("…+", "……", target_sentence)
                target_sentence = re.sub("\.。", "。", target_sentence)
                target_sentence = re.sub("。\.", "。", target_sentence)
                target_sentence = re.sub("\?。", "?", target_sentence)
                target_sentence = re.sub("。\?", "。", target_sentence)
                target_sentence = re.sub("？。", "？", target_sentence)
                target_sentence = re.sub("。？", "。", target_sentence)
                target_sentence = re.sub("，。$", "。", target_sentence)
                target_sentence = re.sub("。，$", "。", target_sentence)
                target_sentence = re.sub(",。$", "。", target_sentence)
                target_sentence = re.sub("。,$", "。", target_sentence)
                target_sentence = converter.convert(target_sentence)
                if target_sentence == source_sentence or len(target_sentence) == 0:
                    continue
                if not re.search('[^=,.?!@#$%^&\*()_+:"<>/\[\]\\`~–—，。、《》？;；‘’：“”【】、{}|·！￥…（）」●＊-]', target_sentence):
                    continue
                if target_sentence not in data[source_sentence]:
                    data[source_sentence].append(target_sentence)

        for source_sentence, target_sentences in data.items():
            if len(target_sentences) == 0:
                continue
            if len(target_sentences) == 1 and (source_sentence == target_sentences[0]):
                continue  # delete samples whose input does not contain grammatical errors
            print("S", source_sentence, sep="\t", file=f_out)
            count_source_sentence += 1
            for target_sentence in target_sentences:
                print("T", target_sentence, sep="\t", file=f_out)
                count_target_sentence += 1
            print(file=f_out)

        assert filter_sentences_lang8.issubset(has_filtered_sentemces), "Incomplete filtering Lang-8!"
        print(f'#filter sentences: {filter_num}')
        print(f'#input: {count_source_sentence} #ref: {count_target_sentence} #input/#ref: {count_target_sentence / count_source_sentence}')
