import re
import pickle
import warnings
from util import seq_language_classifier
from crawl_from_db import PULL_DATA_from_DB
warnings.simplefilter(action='ignore', category=FutureWarning)


class Appannie:
    def __init__(self,appannie_and_asa):
        self.asa_daily_performance = appannie_and_asa['asa_daily_performance']
        self.appannie_app_meta = appannie_and_asa['appannie_app_meta']
        self.appannie_keyword_list = appannie_and_asa['appannie_keyword_list']
        self.appannie_related_apps = appannie_and_asa['appannie_related_apps']
        self.appannie_keyword_performance = \
            appannie_and_asa['appannie_keyword_performance']
        # DATA PROCESS PART
        self._p_id2p_name_()
        self._src_pid2dst_pid_()
        self._p_id2kw_list_()
        self._p_id2kwpf2data_()
        self._p_name2taxonomy_data_and_p_name2p_id_(self.p_id2p_name)
        self._p_id2category_()
        self._p_id2description_()
        self._p_name2company_name_()

    def _load_top100_data_(self, p_id, top100_dataset_filename):
        path = 'sql-dataset/'
        path_top100_dataset_filename = path + top100_dataset_filename
        with open(path_top100_dataset_filename, 'r') as f:
            top100_data = f.readlines()
        top100_name_list = list()
        for name in top100_data[0].split(','):
            name = name.replace('\n', '')
            top100_name_list.append(name)
        top100_ID_list = list()
        for ID in top100_data[1].split(','):
            ID = ID.replace('\n', '')
            top100_ID_list.append(int(ID))
        self.top100_p_id = list()
        for i in range(len(top100_ID_list)):
            if top100_ID_list[i] in self.p_id2p_name and \
                    top100_ID_list[i] in self.p_id2kwpf2data:
                self.top100_p_id.append(top100_ID_list[i])


    def _p_name2company_name_(self):
        self.p_name2company_name = dict()
        product_name_app_meta = self.appannie_app_meta['product_name']
        company_name = self.appannie_app_meta['company_name']
        for i in range(len(product_name_app_meta)):
            p_name = product_name_app_meta[i]
            company_name_i = company_name[i]
            if p_name != 'nan' and p_name != 'None' \
                    and company_name_i != 'nan' and company_name_i != 'None':
                if p_name not in self.p_name2company_name:
                    self.p_name2company_name[p_name] = company_name_i
                else:
                    if company_name_i != self.p_name2company_name[p_name]:
                        print('[ERROR] : one by multi!!')


    def _p_id2description_(self):
        self.p_id2description = dict()
        p_id_app_meta = self.appannie_app_meta['product_id']
        product_name_app_meta = self.appannie_app_meta['product_name']
        product_description_app_meta = self.appannie_app_meta['description']
        for i in range(len(p_id_app_meta)):
            p_id, p_name, p_decrip = \
                str(p_id_app_meta[i]), str(product_name_app_meta[i]),\
                str(product_description_app_meta[i])
            if p_id != 'nan' and p_id != 'None' and \
                    p_name != 'nan' and p_name != 'None' and \
                    p_decrip != 'nan' and p_decrip != 'None':
                if int(p_id) not in self.p_id2description:
                    self.p_id2description[int(p_id)] = [p_decrip, p_name]
                else:
                    p_name_origin = self.p_id2description[int(p_id)][1]

    def _DP_for_decription_(self, text):
        text_sentence = text.split('\n')
        for i in range(len(text_sentence)):
            sentence = \
                re.sub('\W+', ' ', text_sentence[i]).replace("_", '') + '.'
            text_sentence[i] = sentence
        text = ''.join(text_sentence)
        return text

    def _p_id2p_name_(self):
        # product_id -> product_name
        p_id_app_meta = self.appannie_app_meta['product_id']
        product_name_app_meta = self.appannie_app_meta['product_name']
        self.p_id2p_name = dict()
        for i in range(len(p_id_app_meta)):
            if str(p_id_app_meta[i]) != 'nan' and \
                    str(p_id_app_meta[i]) != 'None':
                if int(p_id_app_meta[i]) not in self.p_id2p_name:
                    self.p_id2p_name[int(p_id_app_meta[i])] = \
                        product_name_app_meta[i]
        self.p_id2p_name[1483881028] = '金豹娛樂城'
        self.p_id2p_name[1367319220] = 'Phototile'
        self.p_id2p_name[1449543099] = '86400'
        self.p_id2p_name[1473285922] = 'EverestGold'

    def _src_pid2dst_pid_(self):
        # src_product_id -> dst_p_id
        src_product_id = self.appannie_related_apps['src_product_id']
        dst_product_id = self.appannie_related_apps['dst_product_id']
        self.src_pid2dst_pid = dict()
        for i in range(len(src_product_id)):
            if int(src_product_id[i]) not in self.src_pid2dst_pid:
                self.src_pid2dst_pid[int(src_product_id[i])] = list()
            self.src_pid2dst_pid[int(src_product_id[i])] \
                .append(int(dst_product_id[i]))

    def _p_id2kw_list_(self):
        # product_id -> keyword_list
        p_id_kwl = list(self.appannie_keyword_list['product_id'])
        keyw_kwl = list(self.appannie_keyword_list['keyword_list1'])
        self.p_id2kw_list = dict()
        for i in range(len(p_id_kwl)):
            try:
                p_id_kwl[i] = int(p_id_kwl[i])
            except ValueError:
                a = 0
            else:
                p_id_kwl[i] = int(p_id_kwl[i])
                if p_id_kwl[i] not in self.p_id2kw_list:
                    self.p_id2kw_list[p_id_kwl[i]] = list()
                if isinstance(keyw_kwl[i], str) and len(keyw_kwl[i]) > 0:
                    keyw_kwl_i = keyw_kwl[i].split('\n')
                    self.p_id2kw_list[p_id_kwl[i]] += keyw_kwl_i

    def _p_id2kwpf2data_(self):
        p_id_kwpf = list(self.appannie_keyword_performance['product_id'])
        keyw_kepf = list(self.appannie_keyword_performance['keyword'])
        search_volume = \
            list(self.appannie_keyword_performance['search_volume'])
        difficulty = list(self.appannie_keyword_performance['difficulty'])
        rank = list(self.appannie_keyword_performance['rank'])
        traffic_share = \
            list(self.appannie_keyword_performance['traffic_share'])
        score = list(self.appannie_keyword_performance['score'])
        country = list(self.appannie_keyword_performance['country'])
        for i in range(len(traffic_share)):
            if str(traffic_share[i]) == 'None':
                traffic_share[i] = 0
        # product_id -> keyword -> data(rank,traffic_share,score)
        self.p_id2kwpf2data = dict()
        for i in range(len(p_id_kwpf)):
            if len(keyw_kepf[i]) != 0:
                if int(p_id_kwpf[i]) not in self.p_id2kwpf2data:
                    self.p_id2kwpf2data[int(p_id_kwpf[i])] = dict()
                if keyw_kepf[i] not in self.p_id2kwpf2data[int(p_id_kwpf[i])]:
                    self.p_id2kwpf2data[int(p_id_kwpf[i])][keyw_kepf[i]] = \
                        list()
                if str(search_volume[i]) == 'None':
                    search_volume_i = 0
                else:
                    search_volume_i = float(search_volume[i])
                empty_tocken = 'None'
                rank_i = rank[i]
                country_i = country[i]
                if str(rank_i) != empty_tocken and search_volume_i > 0:
                    rank_i = int(rank[i])
                    traffic_share_i = float(traffic_share[i])
                    self.p_id2kwpf2data[int(p_id_kwpf[i])][keyw_kepf[i]] \
                        .append([rank_i, traffic_share_i, search_volume_i])

    def _p_name2taxonomy_data_and_p_name2p_id_(self, p_id2p_name):
        p_id_list = list(p_id2p_name.keys())
        p_name_list = list(p_id2p_name.values())
        self.p_name2p_id = dict()
        for i in range(len(p_name_list)):
            if p_name_list[i] not in self.p_name2p_id:
                self.p_name2p_id[p_name_list[i]] = p_id_list[i]
            else:
                if p_id_list[i] != self.p_name2p_id[p_name_list[i]]:
                    if p_id_list[i] == 542180079 or \
                            self.p_name2p_id[p_name_list[i]] == 542180079:
                        self.p_name2p_id[p_name_list[i]] = 542180079
                    if p_id_list[i] == 1397351267 or \
                            self.p_name2p_id[p_name_list[i]] == 1397351267:
                        self.p_name2p_id[p_name_list[i]] = 1397351267

    def _p_id2category_(self):
        product_id = list(self.appannie_app_meta['product_id'])
        main_category = list(self.appannie_app_meta['main_category'])
        other_category_paths = \
            list(self.appannie_app_meta['other_category_paths'])
        self.p_id2category = dict()
        for i in range(len(product_id)):
            # main_category PART
            if len(main_category[i]) == 0:
                main_category[i] == 'None1'
            main_category_i = [main_category[i]]
            # other_category PART
            other_category_i = list()
            if isinstance(other_category_paths[i], str) and \
                    len(other_category_paths[i]) > 0:
                other_category_paths_i_split = \
                    other_category_paths[i].split(',')
                for j in range(len(other_category_paths_i_split)):
                    other_category_paths_i_split_split = \
                        other_category_paths_i_split[j].split('>')
                    if other_category_paths_i_split_split[-1] != 'Overall':
                        other_category_i \
                            .append(other_category_paths_i_split_split[-1])
            category_i = main_category_i + other_category_i
            if int(product_id[i]) not in self.p_id2category:
                self.p_id2category[int(product_id[i])] = category_i

    def Search_Terms_For_Self_and_Related(self, p_id,
                                          target_restrict_rank_TOP=5,
                                          source_restrict_rank_TOP=5):
        # Target PART
        if p_id in self.p_id2kwpf2data:
            target_R_list, target_TS_list, target_OTHER_list = \
                self.R_TS_filter_layer(p_id, target_restrict_rank_TOP)
            target_R_TS_list = list(set(target_R_list) | set(target_TS_list))
        else:
            target_R_TS_list = None
        if p_id in self.p_id2p_name:
            target_name = self.p_id2p_name[p_id]
        else:
            target_name = None
        target_part = [target_name, target_R_TS_list]
        # Source PART
        source_part = []
        if p_id not in self.src_pid2dst_pid:
            dst_pid_list = None
        else:
            dst_pid_list = self.src_pid2dst_pid[p_id]
            for i in range(len(dst_pid_list)):
                if dst_pid_list[i] in self.p_id2p_name:
                    source_name = self.p_id2p_name[dst_pid_list[i]]
                else:
                    source_name = None
                if dst_pid_list[i] in self.p_id2kwpf2data:
                    source_R_list, source_TS_list, source_OTHER_list = \
                        self.R_TS_filter_layer(dst_pid_list[i],
                                               source_restrict_rank_TOP)
                    source_R_TS_list = \
                        list(set(source_R_list) | set(source_TS_list))
                else:
                    source_R_TS_list = None
                source_part.append([source_name, source_R_TS_list])
        return target_part, source_part

    def Search_Terms_For_Self_and_TOP100(self, p_id, top100_p_id,
                                         target_restrict_rank_TOP=5,
                                         top100_restrict_rank_TOP=5):
        # Target PART
        if p_id in self.p_id2kwpf2data:
            target_R_list, target_TS_list, target_OTHER_list = \
                self.R_TS_filter_layer(p_id, target_restrict_rank_TOP)
            target_R_TS_list = list(set(target_R_list) | set(target_TS_list))
        else:
            target_R_TS_list = None
        if p_id in self.p_id2p_name:
            target_name = self.p_id2p_name[p_id]
        else:
            target_name = None
        target_part = [target_name, target_R_TS_list]

        # Top100 PART
        Top100_part, Top100_pid2TS_keyw = list(), dict()
        for i in range(len(top100_p_id)):
            p_name = self.p_id2p_name[top100_p_id[i]]
            R_list, TS_list, OTHER_list = \
                self.R_TS_filter_layer(top100_p_id[i],
                                       top100_restrict_rank_TOP)
            R_TS_list = list(set(R_list) | set(TS_list))
            TS_list = list(set(TS_list))
            Top100_part.append([p_name, R_TS_list])
            Top100_pid2TS_keyw[top100_p_id[i]] = TS_list
        return target_part, Top100_part, Top100_pid2TS_keyw

    def R_TS_filter_layer(self, p_id, rank_TOP=5):
        R_list, TS_list, OTHER_list = list(), list(), list()
        if p_id in self.p_id2kwpf2data:
            kwpf2data = self.p_id2kwpf2data[p_id]
            keyw_list = list(kwpf2data.keys())
            for i in range(len(keyw_list)):
                data_list = kwpf2data[keyw_list[i]]
                if len(data_list) != 0:
                    data = data_list[0]
                    rank_i = data[0]
                    traffic_share_i = data[1]
                    score_i = data[2]
                    # Long tail word filter
                    if traffic_share_i > 0:
                        TS_list.append(keyw_list[i])
                    if rank_i <= rank_TOP:
                        R_list.append(keyw_list[i])
                    if traffic_share_i <= 0 or rank_i > rank_TOP:
                        OTHER_list.append(keyw_list[i])
            R_list = list(set(R_list))
            TS_list = list(set(TS_list))
            OTHER_list = list(set(OTHER_list))
            return R_list, TS_list, OTHER_list
        else:
            return [], [], []


def _app_organic_recall_from_appannie_(appannie_and_asa, config):
    # config go to here
    target_app_id = config.to('target_app_id')
    rank_of_target_app = config.to('rank_of_target_app')
    rank_of_top100_app = config.to('rank_of_top100_app')
    rank_of_related_app = config.to('rank_of_related_app')
    top100_filename = config.to('top100_filename')
    # using Appannie object
    appannie = Appannie(appannie_and_asa)
    # top100 id
    appannie._load_top100_data_(target_app_id, top100_filename)
    top100_app_id = appannie.top100_p_id
    # Recall organic from appannie (target app part and top-100 part)
    target_part, top100_part, _ = \
        appannie.Search_Terms_For_Self_and_TOP100(target_app_id,
                                                  top100_app_id,
                                                  rank_of_target_app,
                                                  rank_of_top100_app)
    # Recall organic from appannie (related app part)
    _, related_app_part = \
        appannie.Search_Terms_For_Self_and_Related(target_app_id,
                                                   rank_of_target_app,
                                                   rank_of_related_app)
    # app description part
    if len(top100_part) > 0:
        top100_app_name = [top100_data[0] for top100_data in top100_part]
    else:
        top100_app_name = []
    if len(related_app_part) > 0 :
        related_app_name = [related_app_data[0] for related_app_data in related_app_part]
    else:
        related_app_name = []
    app_name_list = list(set([target_part[0]] + top100_app_name + related_app_name))
    app_name2description = dict()
    app_name2main_used_language = dict()
    p_name2p_id = appannie.p_name2p_id
    p_id2description = appannie.p_id2description
    for i in range(len(app_name_list)):
        p_id = p_name2p_id[app_name_list[i]]
        description = p_id2description[p_id][0]
        main_used_language = \
            seq_language_classifier(description.split('\n'))
        app_name2main_used_language[app_name_list[i]] = \
            main_used_language
        description_DP = appannie._DP_for_decription_(description)
        app_name2description[app_name_list[i]] = description_DP
    # app -> organic_neighbor
    # corpus collection(universal_set | candidate_keyw)
    app_name2neighbor, universal_set, candidate_keyw = dict(), list(), list()
    # target part
    target_app_name = target_part[0]
    target_app_main_ln = \
        app_name2main_used_language[target_app_name]
    app_name2neighbor[target_app_name] = target_part[1]
    universal_set += [target_app_name]
    universal_set += target_part[1]
    candidate_keyw += target_part[1]
    # top100 part
    top100_app_name = list()
    for i in range(len(top100_part)):
        if len(top100_part[i][1]) > 0:
            if top100_part[i][0] not in app_name2neighbor:
                # top100_app_main_ln = \
                #     app_name2main_used_language[top100_part[i][0]]
                #if top100_app_main_ln == target_app_main_ln:
                top100_app_name.append(top100_part[i][0])
                app_name2neighbor[top100_part[i][0]] = \
                    top100_part[i][1]
                universal_set += top100_part[i][1]
                universal_set += [top100_part[i][0]]
                candidate_keyw += top100_part[i][1]
    # related app part
    related_app_name = list()
    for i in range(len(related_app_part)):
        if len(related_app_part[i][1]) > 0:
            if related_app_part[i][0] not in app_name2neighbor:
                # related_app_main_ln = \
                #     app_name2main_used_language[related_app_part[i][0]]
                #if related_app_main_ln == target_app_main_ln:
                related_app_name.append(related_app_part[i][0])
                app_name2neighbor[related_app_part[i][0]] = \
                    related_app_part[i][1]
                universal_set += [related_app_part[i][0]]
                universal_set += related_app_part[i][1]
                candidate_keyw += related_app_part[i][1]
    # other_app_name = top100_app_name + related_app_name
    other_app_name = list(set(top100_app_name) | set(related_app_name))

    # corpus part (for W2V)
    universal_set = list(set(universal_set))
    # candidate keyword (for prediciton result)
    candidate_keyw = list(set(candidate_keyw))
    # keyword -> app_name (for DA of prediction result)
    keyw2app_name = dict()
    app_name_list = list(app_name2neighbor.keys())
    for i in range(len(app_name_list)):
        neighbor = app_name2neighbor[app_name_list[i]]
        for j in range(len(neighbor)):
            if neighbor[j] not in keyw2app_name:
                keyw2app_name[neighbor[j]] = set()
            keyw2app_name[neighbor[j]].add(app_name_list[i])
    # app_name -> company_name
    app_name2company_name = appannie.p_name2company_name
    company_name_list = set()
    for app_name in app_name_list:
        if app_name in app_name2company_name:
            company_name = app_name2company_name[app_name]
            company_name_list.add(company_name)
        else:
            print('app_name:',app_name)

    # go to config
    function_name = '_app_organic_recall_from_appannie_'
    config.to_config(data=appannie,
                     data_name='appannie',
                     come_from=function_name)
    config.to_config(data=app_name2neighbor,
                     data_name='app_name2neighbor',
                     come_from=function_name)
    config.to_config(data=universal_set,
                     data_name='universal_set',
                     come_from=function_name)
    config.to_config(data=candidate_keyw,
                     data_name='candidate_keyw',
                     come_from=function_name)
    config.to_config(data=target_app_name,
                     data_name='target_app_name',
                     come_from=function_name)
    config.to_config(data=top100_app_name,
                     data_name='top100_app_name',
                     come_from=function_name)
    config.to_config(data=related_app_name,
                     data_name='related_app_name',
                     come_from=function_name)
    config.to_config(data=other_app_name,
                     data_name='other_app_name',
                     come_from=function_name)
    config.to_config(data=keyw2app_name,
                     data_name='keyw2app_name',
                     come_from=function_name)
    config.to_config(data=app_name2description,
                     data_name='app_name2description',
                     come_from=function_name)
    config.to_config(data=app_name2company_name,
                     data_name='app_name2company_name',
                     come_from=function_name)
    return config
