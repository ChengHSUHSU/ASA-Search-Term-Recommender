import os
import re
from opencc import OpenCC
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as Cosin_Distance
import requests
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def Cosine_Similarity(main_node_embedding, other_embedding_list,
                      other_name_list, TOP_N=15):
    # calculating CS of main node and other node ans sorted
    main_node_and_pos_keyword_embedding = \
        [main_node_embedding] + other_embedding_list
    main_node_and_pos_keyword_embedding_array = \
        np.array(main_node_and_pos_keyword_embedding)
    CS_matrix_array = Cosin_Distance(main_node_and_pos_keyword_embedding_array)
    csm_list = list()
    for i in range(CS_matrix_array.shape[1] - 1):
        csm_list.append((CS_matrix_array[0, i + 1], other_name_list[i]))
    csm_list_sorted = sorted(csm_list, reverse=True)
    csm_list_sorted_TOP = csm_list_sorted[: TOP_N]
    return csm_list_sorted_TOP


def detect_traditional_ch(text):
    ch_is_tradition = True
    cc = OpenCC('s2tw')
    text_convert = cc.convert(text)
    if text != text_convert:
        ch_is_tradition = False
    return ch_is_tradition


def language_classifier(text):
    # ONLY support traditional/simple chinese | Japanese | Korean | English
    # traditional/simple chinese
    re_words_ch = re.compile(u"[\u4e00-\u9fa5]+")
    res_ch = re.findall(re_words_ch, text)
    # Japanese
    re_words_jp = re.compile(u"[\u30a0-\u30ff\u3040-\u309f]+")
    res_jp = re.findall(re_words_jp, text)
    # korean
    re_words_ko = re.compile(u"[\uac00-\ud7ff]+")
    res_ko = re.findall(re_words_ko, text)
    # English
    re_words_en = re.compile(u"[a-zA-Z]")
    res_en = re.findall(re_words_en, text)

    part_word_ch, part_word_jp, part_word_ko = '', '', ''
    if len(res_ch) > 0:
        part_word_ch = ''.join(res_ch)
    if len(res_jp) > 0:
        part_word_jp = ''.join(res_jp)
    if len(res_ko) > 0:
        part_word_ko = ''.join(res_ko)
    used_language = list()
    if len(part_word_ch) > 0 and \
            len(part_word_jp) == 0 and \
            len(part_word_ko) == 0:
        if detect_traditional_ch(text):
            used_language.append('traditional_ch')
        else:
            used_language.append('simple_ch')
    if len(part_word_jp) > 0 and len(part_word_ko) == 0:
        used_language.append('jp')
    if len(part_word_ch) == 0 and \
            len(part_word_jp) == 0 and \
            len(part_word_ko) > 0:
        used_language.append('ko')
    if len(part_word_ch) == 0 and \
            len(part_word_jp) == 0 and \
            len(part_word_ko) == 0:
        if len(res_en) > 0:
            used_language.append('en')
    used_language = list(set(used_language))
    return used_language


def seq_language_classifier(seq_text):
    language2used_num = dict()
    for i in range(len(seq_text)): 
        used_ln = language_classifier(seq_text[i])
        if len(used_ln) == 1:
            if used_ln[0] not in language2used_num:
                language2used_num[used_ln[0]] = 0
            language2used_num[used_ln[0]] +=1
    ln_list = list(language2used_num.keys())
    used_num_and_ln = [(language2used_num[ln],ln) for ln in ln_list]
    seq_main_used_language = max(used_num_and_ln)[1]
    return seq_main_used_language




def forecasting_result_TO_endpoint(forecasting_result, path):
    try:
        os.remove(path)
    except FileNotFoundError:
        a = 0
    with open(path, 'w') as f:
        for i in range(len(forecasting_result)):
            if i != len(forecasting_result) -1:
                f.write(forecasting_result[i] + ',')
            else:
                f.write(forecasting_result[i])


def crawl_sub_KG(keyword):
    used_language = language_classifier(keyword)
    if len(used_language) == 1:
        if used_language[0] == 'traditional_ch' or used_language[0] == 'simple_ch':
            ln = 'zh'
        else:
            ln = used_language[0]
        url = 'http://api.conceptnet.io/c/' + ln + '/' + keyword + '?limit=9999999999'
        obj = requests.get(url).json()
        return obj
    else:
        return None