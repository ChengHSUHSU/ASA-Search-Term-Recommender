import re
import pyprind
import pandas as pd
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as Cosin_Distance
import pickle
from transformers import AutoTokenizer
from transformers import AutoModel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class ASA_Daily_Data:
    def __init__(self):
        # DATA LOAD
        pickle_path = 'sql-dataset/' + 'appannie_and_asa_data.pickle'
        with open(pickle_path, 'rb') as file:
            a_dict1 = pickle.load(file)
            appannie_and_asa = a_dict1['appannie_and_asa']
        asa_daily_performance = appannie_and_asa['asa_daily_performance']
        self.campaign_name = asa_daily_performance['campaign_name']
        self.keyword = asa_daily_performance['keyword']
        self.impressions = asa_daily_performance['impressions']
        self.clicks = asa_daily_performance['clicks']
        self.installs = asa_daily_performance['installs']
        self.local_spend = asa_daily_performance['local_spend']
        self.date = asa_daily_performance['date']
        self.crawled_date = asa_daily_performance['crawled_time']
        self.account_id_list = asa_daily_performance['account_id']
        asa_hourly_record = appannie_and_asa['asa_hourly_record']
        self.account_id2mode2date2keyw2performance = dict()

    def _account_id2keyw2performance_(self, account_id):
        account_id = int(account_id)
        if account_id not in self.account_id2keyw2performance:
            self.account_id2keyw2performance[account_id] = dict()
            for i in range(len(self.account_id_list)):
                if account_id == int(self.account_id_list[i]):
                    keyword_i = self.keyword[i]
                    if str(keyword_i) != 'nan' and str(keyword_i) != 'None':
                        impressions_i = int(self.impressions[i])
                        clicks_i = int(self.clicks[i])
                        installs_i = int(self.installs[i])
                        local_spend_i = int(self.local_spend[i])
                        if keyword_i not in \
                                self.account_id2keyw2performance[account_id]:
                                    self.account_id2keyw2performance[
                                        account_id][keyword_i] = \
                                        {'impressions': impressions_i,
                                         'clicks': clicks_i,
                                         'installs': installs_i,
                                         'local_spend': local_spend_i}
                        else:
                            self.account_id2keyw2performance[
                                account_id][keyword_i]['impressions'] \
                                += impressions_i
                            self.account_id2keyw2performance[
                                account_id][keyword_i]['clicks'] \
                                += clicks_i
                            self.account_id2keyw2performance[
                                account_id][keyword_i]['installs'] \
                                += installs_i
                            self.account_id2keyw2performance[
                                account_id][keyword_i]['local_spend'] \
                                += local_spend_i

    def _account_id2mode2date2keyw2performance_(self, account_id,campaign_setting=None):
        if campaign_setting is None:
            campaign_setting = self.campaign_name
        if account_id not in self.account_id2mode2date2keyw2performance:
            self.account_id2mode2date2keyw2performance[account_id] = dict()
            self.account_id2mode2date2keyw2performance[account_id] = \
                {'overall': dict(),
                 'date': dict()}
            for i in range(len(self.account_id_list)):
                if account_id == int(self.account_id_list[i]) and self.campaign_name[i] in campaign_setting:
                    overall_mode = \
                        self.account_id2mode2date2keyw2performance[
                            account_id]['overall']
                    date_mode = \
                        self.account_id2mode2date2keyw2performance[
                            account_id]['date']
                    date_i = str(self.date[i])
                    keyword_i = self.keyword[i]
                    if date_i not in date_mode:
                        date_mode[date_i] = dict()
                    if str(keyword_i) != 'nan' and str(keyword_i) != 'None':
                        impressions_i = int(self.impressions[i])
                        clicks_i = int(self.clicks[i])
                        installs_i = int(self.installs[i])
                        local_spend_i = int(self.local_spend[i])
                        if keyword_i not in date_mode[date_i]:
                            date_mode[date_i][keyword_i] = \
                                {'impressions': impressions_i,
                                 'clicks': clicks_i,
                                 'installs': installs_i,
                                 'local_spend': local_spend_i}
                        else:
                            date_mode[date_i][keyword_i]['impressions'] \
                                += impressions_i
                            date_mode[date_i][keyword_i]['clicks'] \
                                += clicks_i
                            date_mode[date_i][keyword_i]['installs'] \
                                += installs_i
                            date_mode[date_i][keyword_i]['local_spend'] \
                                += local_spend_i
                    if str(keyword_i) != 'nan' and str(keyword_i) != 'None':
                        impressions_i = int(self.impressions[i])
                        clicks_i = int(self.clicks[i])
                        installs_i = int(self.installs[i])
                        local_spend_i = int(self.local_spend[i])
                        if keyword_i not in overall_mode:
                            overall_mode[keyword_i] = \
                                {'impressions': impressions_i,
                                 'clicks': clicks_i,
                                 'installs': installs_i,
                                 'local_spend': local_spend_i}
                        else:
                            overall_mode[keyword_i]['impressions'] \
                                += impressions_i
                            overall_mode[keyword_i]['clicks'] \
                                += clicks_i
                            overall_mode[keyword_i]['installs'] \
                                += installs_i
                            overall_mode[keyword_i]['local_spend'] \
                                += local_spend_i
                    self.account_id2mode2date2keyw2performance[
                        account_id]['overall'] = overall_mode
                    self.account_id2mode2date2keyw2performance[
                        account_id]['date'] = date_mode

    def TEST_FIND_POS(self, account_id, determine_train_date, 
                      determine_test_date,campaign_setting, mode='overall'):
        self._account_id2mode2date2keyw2performance_(account_id,
                                                     campaign_setting)
        mode2date2keyw2performance = \
            self.account_id2mode2date2keyw2performance[account_id]
        # overall
        if mode == 'overall':
            keyw2performance = mode2date2keyw2performance['overall']
            keyw_list = list(keyw2performance.keys())
            self.impression_pos_keyw = list()
            self.click_pos_keyw = list()
            self.install_pos_keyw = list()
            install_keyw_and_num = list()
            for i in range(len(keyw_list)):
                performance = keyw2performance[keyw_list[i]]
                impressions = performance['impressions']
                clicks = performance['clicks']
                installs = performance['installs']
                if impressions > 0:
                    self.impression_pos_keyw.append(keyw_list[i])
                if clicks > 0:
                    self.click_pos_keyw.append(keyw_list[i])
                if installs > 0:
                    self.install_pos_keyw.append(keyw_list[i])
                    install_keyw_and_num.append((installs, keyw_list[i]))
            self.impression_pos_keyw = list(set(self.impression_pos_keyw))
            self.click_pos_keyw = list(set(self.click_pos_keyw))
            self.install_pos_keyw = list(set(self.install_pos_keyw))
            install_keyw_and_num = \
                sorted(install_keyw_and_num, reverse=True)
            self.install_keyw_rank = list()
            for keyw_and_num in install_keyw_and_num:
                if keyw_and_num[1] not in self.install_keyw_rank:
                    self.install_keyw_rank.append(keyw_and_num[1])
            return self.impression_pos_keyw, self.click_pos_keyw, \
                self.install_pos_keyw, self.install_keyw_rank
        elif mode == 'date':
            date2keyw2performance = mode2date2keyw2performance['date']
            if determine_train_date is not None and determine_test_date is not None:
                date_list = list(date2keyw2performance.keys())
                train_start_time = determine_train_date[0]
                train_end_time = determine_train_date[1]
                test_start_time = determine_test_date[0]
                test_end_time = determine_test_date[1]
                if train_start_time is None:
                    train_start_time = min(date_list)
                if train_end_time is None:
                    train_end_time = max(date_list)
                if test_start_time is None:
                    test_start_time = min(date_list)
                if test_end_time is None:
                    test_end_time = max(date_list)
                impression_train, click_train, install_train = \
                    list(), list(), list()
                impression_test, click_test, install_test = \
                    list(), list(), list()
                install_keyw_and_num = list()
                for i in range(len(date_list)):
                    keyw2performance = date2keyw2performance[date_list[i]]
                    keyw_list = list(keyw2performance.keys())
                    impression_date_j = list()
                    click_date_j = list()
                    install_date_j = list()
                    for j in range(len(keyw_list)):
                        performance = keyw2performance[keyw_list[j]]
                        impressions = performance['impressions']
                        clicks = performance['clicks']
                        installs = performance['installs']
                        if impressions > 0:
                            impression_date_j.append(keyw_list[j])
                        if clicks > 0:
                            click_date_j.append(keyw_list[j])
                        if installs > 0:
                            install_date_j.append(keyw_list[j])
                            install_keyw_and_num\
                                .append([installs, keyw_list[j]])
                    if test_start_time <=  date_list[i] and date_list[i] <= test_end_time:
                        impression_test += impression_date_j
                        click_test += click_date_j
                        install_test += install_date_j
                    elif train_start_time <= date_list[i] and date_list[i] <= train_end_time:
                        impression_train += impression_date_j
                        click_train += click_date_j
                        install_train += install_date_j
                impression_train = list(set(impression_train))
                click_train = list(set(click_train))
                install_train = list(set(install_train))
                impression_test = \
                    list(set(impression_test) - set(impression_train))
                click_test = \
                    list(set(click_test) - set(impression_train))
                install_test = \
                    list(set(install_test) - set(impression_train))
                install_keyw_and_num = \
                    sorted(install_keyw_and_num, reverse=True)
                install_keyw_rank = list()
                for num_and_keyw in install_keyw_and_num:
                    if num_and_keyw[1] not in install_keyw_rank:
                        install_keyw_rank.append(num_and_keyw[1])
                return impression_train, click_train, install_train, \
                    impression_test, click_test, install_test, \
                    install_keyw_rank

    def TEST_POS_Keyword_Embedding(self, target_part, pos_keyw_list):
        main_node_name = target_part[0]
        pos_keyw_list = [main_node_name] + pos_keyw_list
        self.tokenizer = \
            AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        self.pos_keyw2embedding = dict()
        for i in pyprind.prog_bar(range(len(pos_keyw_list))):
            word_tocken = self.tokenizer(pos_keyw_list[i], return_tensors="pt")
            word_vector = \
                self.model(**word_tocken)[1].cpu().detach().numpy().reshape(-1)
            self.pos_keyw2embedding[pos_keyw_list[i]] = list(word_vector)

    def TEST_Cos_Sim_Rank(self, target_part, pos_keyword_list,
                          source_part_i=None, TOP_N=10):
        main_node_name = target_part[0]
        main_node_and_pos_keyword_list = [main_node_name] + pos_keyword_list
        main_node_and_pos_keyword_embedding = list()
        for i in range(len(main_node_and_pos_keyword_list)):
            main_node_and_pos_keyword_embedding.append(
                self.pos_keyw2embedding[main_node_and_pos_keyword_list[i]])
        main_node_and_pos_keyword_embedding_array = \
            np.array(main_node_and_pos_keyword_embedding)
        CS_matrix_array = \
            Cosin_Distance(main_node_and_pos_keyword_embedding_array)
        csm_list = list()
        for i in range(CS_matrix_array.shape[1]-1):
            csm_list.append((CS_matrix_array[0, i+1], pos_keyword_list[i]))
        csm_list_sorted = sorted(csm_list, reverse=True)
        csm_list_sorted_TOP = csm_list_sorted[: TOP_N]
        return csm_list_sorted_TOP

    def TEST_CS_Baseline(self, target_part, impression_pos_keyw,
                         click_pos_keyw, TOP_N=15):
        impression_part = \
            self.TEST_Cos_Sim_Rank(target_part,
                                   impression_pos_keyw,
                                   None, TOP_N)
        impression_part_keyw_list = \
            [impression_part[i][1] for i in range(len(impression_part))]
        click_part_keyw_list = None
        return impression_part_keyw_list, click_part_keyw_list

    def TEST_Random_Baseline(self, target_part, impression_pos_keyw,
                             click_pos_keyw, TOP_N=15):
        impression_pos_keyw = list(set(impression_pos_keyw))
        impression_pos_keyw_random = \
            random.sample(impression_pos_keyw, len(impression_pos_keyw))
        impression_part_keyw_list = impression_pos_keyw_random[: TOP_N]
        return impression_part_keyw_list


def _keyw2performance_based_on_date_(config, asa_daily_data,
                                     asa_account_id, 
                                     determine_interval_date):
    # config come here
    determine_start_date = determine_interval_date[0]
    determine_end_date = determine_interval_date[1]
    account_id2mode2date2keyw2performance = \
        asa_daily_data.account_id2mode2date2keyw2performance
    date2keyw2performance = \
        account_id2mode2date2keyw2performance[asa_account_id]['date']
    keyw2performance_based_on_date = dict()
    date_list = list(date2keyw2performance.keys())

    if determine_start_date is None:
        determine_start_date = min(date_list)
    if determine_end_date is None:
        determine_end_date = max(date_list)
    for date in date_list:
        if date <= determine_end_date and determine_start_date <= date:
            keyw2performance = date2keyw2performance[date]
            keyw_list = list(keyw2performance.keys())
            for keyw in keyw_list:
                performance = keyw2performance[keyw]
                impressions = performance['impressions']
                clicks = performance['clicks']
                installs = performance['installs']
                local_spend = performance['local_spend']
                if keyw not in keyw2performance_based_on_date:
                    keyw2performance_based_on_date[keyw] = \
                        {'impressions': 0,
                        'clicks': 0,
                        'installs': 0,
                        'local_spend':0}
                keyw2performance_based_on_date[keyw]['impressions'] += \
                    impressions
                keyw2performance_based_on_date[keyw]['clicks'] += \
                    clicks
                keyw2performance_based_on_date[keyw]['installs'] += \
                    installs
                keyw2performance_based_on_date[keyw]['local_spend'] += \
                    local_spend
    return keyw2performance_based_on_date


def _online_data_recall_from_asa_(show_DA_for_Data, config):
    # config go to here
    asa_account_id = config.to('asa_account_id')
    candidate_keyw = config.to('candidate_keyw')
    universal_set = config.to('universal_set')
    determine_train_date = config.to('determine_train_date')
    determine_test_date = config.to('determine_test_date')
    determine_forecast_date = config.to('determine_forecast_date')
    campaign_setting = config.to('campaign_setting')
    # using ASA_Daily_Data object
    asa_daily_data = ASA_Daily_Data()
    # recall asa data based on eval_mode_time
    impression_train, click_train, install_train, \
        impression_test, click_test, install_test, \
        install_keyw_rank = \
        asa_daily_data.TEST_FIND_POS(asa_account_id,
                                     determine_train_date,
                                     determine_test_date,
                                     campaign_setting,
                                     mode='date')
    install_test_in_appannie = \
        list(set(install_test) | set())
    universal_set += impression_train
    universal_set += impression_test
    universal_set += install_train
    universal_set += install_test
    # base on determine date
    keyw2performance_for_determine_date = \
        _keyw2performance_based_on_date_(config,
                                         asa_daily_data,
                                         asa_account_id,
                                         determine_train_date)
    keyw_for_determine_date = list(keyw2performance_for_determine_date.keys())
    # base on overall
    keyw2performance_for_interval = \
        _keyw2performance_based_on_date_(config,
                                         asa_daily_data,
                                         asa_account_id,
                                         determine_forecast_date)
    keyw_for_interval = list(keyw2performance_for_interval.keys())
    universal_set += keyw_for_determine_date
    universal_set += keyw2performance_for_interval
    universal_set = list(set(universal_set))

    keyw2app_name = config.to('keyw2app_name')
    app_name2company_name = config.to('app_name2company_name')
    if show_DA_for_Data:
        impression_non_zero_keyw_num = \
            len(set(impression_test))
        install_non_zero_keyw_num = \
            len(set(install_test))
        impression_non_zero_keyw_num_in_appannie = \
            len(( set(impression_test)) & set(candidate_keyw))
        install_non_zero_keyw_num_in_appannie = \
            len((set(install_test)) & set(candidate_keyw))
        print('impression_train : ',len(impression_train))
        print('install_train : ',len(install_train))
        print('impression_test : ',len(impression_test))
        print('install_test : ',len(install_test))
        print('impression_test | appannie = ',impression_non_zero_keyw_num_in_appannie)
        print('install_test| appannie = ',install_non_zero_keyw_num_in_appannie)
        print('organic num:',len(candidate_keyw))
    
    # go to config
    function_name = '_online_data_recall_from_asa_'
    config.to_config(data=universal_set,
                     data_name='universal_set',
                     come_from=function_name)
    config.to_config(data=asa_daily_data,
                     data_name='asa_daily_data',
                     come_from=function_name)
    config.to_config(data=install_keyw_rank,
                     data_name='install_keyw_rank',
                     come_from=function_name)
    config.to_config(data=keyw2performance_for_determine_date,
                     data_name='keyw2performance_for_determine_date',
                     come_from=function_name)
    config.to_config(data=keyw2performance_for_interval,
                     data_name='keyw2performance_for_interval',
                     come_from=function_name)
    config.to_config(data=impression_train,
                     data_name='impression_train',
                     come_from=function_name)
    config.to_config(data=click_train,
                     data_name='click_train',
                     come_from=function_name)
    config.to_config(data=install_train,
                     data_name='install_train',
                     come_from=function_name)
    config.to_config(data=impression_test,
                     data_name='impression_test',
                     come_from=function_name)
    config.to_config(data=click_test,
                     data_name='click_test',
                     come_from=function_name)
    config.to_config(data=install_test,
                     data_name='install_test',
                     come_from=function_name)
    config.to_config(data=install_test_in_appannie,
                     data_name='install_test_in_appannie',
                     come_from=function_name)
    return config
