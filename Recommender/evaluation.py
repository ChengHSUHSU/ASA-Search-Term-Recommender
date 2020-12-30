import matplotlib.pyplot as plt
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Evaluation:
    def __init__(self, appannie_and_asa):
        asa_daily_performance = appannie_and_asa['asa_daily_performance']
        self.campaign_name = asa_daily_performance['campaign_name']
        self.keyword = asa_daily_performance['keyword']
        self.impressions = asa_daily_performance['impressions']
        self.clicks = asa_daily_performance['clicks']
        self.installs = asa_daily_performance['installs']
        self.local_spend = asa_daily_performance['local_spend']
        self.account_id_list = asa_daily_performance['account_id']
        self.date = asa_daily_performance['date']
        # Building Container
        self.account_id2keyw2performance = dict()
        self.account_id2mode2date2keyw2performance = dict()

    def _account_id2keyw2performance_(self, account_id):
        account_id = int(account_id)
        if account_id not in self.account_id2keyw2performance:
            self.account_id2keyw2performance[account_id] = dict()
            for i in range(len(self.account_id_list)):
                if account_id == int(self.account_id_list[i]):
                    keyword_i = self.keyword[i]
                    if str(keyword_i) != 'nan':
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
                                account_id][keyword_i]['impressions']\
                                += self.impressions[i]
                            self.account_id2keyw2performance[
                                account_id][keyword_i]['clicks']\
                                += self.clicks[i]
                            self.account_id2keyw2performance[
                                account_id][keyword_i]['installs']\
                                += self.installs[i]
                            self.account_id2keyw2performance[
                                account_id][keyword_i]['local_spend'] \
                                += self.local_spend[i]

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
                            date_mode[date_i][keyword_i]['impressions']\
                                += impressions_i
                            date_mode[date_i][keyword_i]['clicks']\
                                += clicks_i
                            date_mode[date_i][keyword_i]['installs']\
                                += installs_i
                            date_mode[date_i][keyword_i]['local_spend']\
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
                            overall_mode[keyword_i]['impressions']\
                                += impressions_i
                            overall_mode[keyword_i]['clicks']\
                                += clicks_i
                            overall_mode[keyword_i]['installs']\
                                += installs_i
                            overall_mode[keyword_i]['local_spend']\
                                += local_spend_i
                    self.account_id2mode2date2keyw2performance[
                        account_id]['overall'] = overall_mode
                    self.account_id2mode2date2keyw2performance[
                        account_id]['date'] = date_mode

    def date2keyw2performanceTOkeyw2performance(self, date2keyw2performance,
                                                determine_test_date):
        start_date = determine_test_date[0]
        end_date = determine_test_date[1]
        keyw2performance = dict()
        date_list = list(date2keyw2performance.keys())
        if start_date is None:
            start_date = min(date_list)
        if end_date is None:
            end_date = max(date_list)
        for i in range(len(date_list)):
            if date_list[i] >= start_date and end_date >= date_list[i]:
                keyw2performance_in_date_i = \
                    date2keyw2performance[date_list[i]]
                keyw_list = list(keyw2performance_in_date_i.keys())
                for j in range(len(keyw_list)):
                    if keyw_list[j] not in keyw2performance:
                        keyw2performance[keyw_list[j]] = \
                            {'installs': 0,
                             'local_spend': 0,
                             'impressions': 0,
                             'clicks': 0}
                    installs = \
                        keyw2performance_in_date_i[keyw_list[j]]['installs']
                    local_spend = \
                        keyw2performance_in_date_i[keyw_list[j]]['local_spend']
                    impressions = \
                        keyw2performance_in_date_i[keyw_list[j]]['impressions']
                    clicks = \
                        keyw2performance_in_date_i[keyw_list[j]]['clicks']
                    keyw2performance[keyw_list[j]]['installs'] += installs
                    keyw2performance[keyw_list[j]]['local_spend'] \
                        += local_spend
                    keyw2performance[keyw_list[j]]['impressions'] \
                        += impressions
                    keyw2performance[keyw_list[j]]['clicks'] += clicks
        return keyw2performance

    def TEST(self, account_id, result_keyw_list, ground_true, config):
        # go to config
        campaign_setting = config.to('campaign_setting')
        determine_test_date = config.to('determine_test_date')
        # determine topK (auto)
        expected_step_num = 6
        if len(ground_true) < 18:
            topk = [1,5,10,15]
        else:
            topK_interval = int(len(ground_true) / expected_step_num)
            topk = [1] + [(i + 1) * topK_interval for i in range(int(len(ground_true) / topK_interval))]
        print('used_topK : ',topk)
        result_keyw_list = result_keyw_list[:topk[-1]]
        # get asa data
        self._account_id2mode2date2keyw2performance_(account_id,campaign_setting)
        result_performance_list = list()
        date2keyw2performance = \
            self.account_id2mode2date2keyw2performance[account_id]['date']
        keyw2performance = \
            self.date2keyw2performanceTOkeyw2performance(date2keyw2performance,
                                                         determine_test_date)
        # calculating #install / cpa and rankly data(cpa/install)
        metric1_spend = 0
        metric1_install = 0
        keyw2pos_click_data = dict()
        rankly_record_for_install_cpa = list()
        rankly_record_for_click = list()
        for keyw in result_keyw_list:
            if keyw in keyw2performance:
                performance = keyw2performance[keyw]
                installs = performance['installs']
                local_spend = performance['local_spend']
                clicks = performance['clicks']
                metric1_spend += local_spend
                metric1_install += installs
                if clicks > 0:
                    keyw2pos_click_data[keyw] = [clicks,installs,local_spend]
                if installs == 0:
                    if clicks > 4:
                        rankly_record_for_click.append([keyw,clicks,installs,local_spend])
                else:
                    keyw_cpa = local_spend / installs
                    if installs / clicks  <  0.25 :
                        rankly_record_for_click.append([keyw,clicks,installs,keyw_cpa])
                    else:
                        rankly_record_for_install_cpa.append([keyw,clicks,installs,keyw_cpa])
        if metric1_install != 0:
            cpa = metric1_spend / metric1_install
        else:
            cpa = 'None'
        print('============================')
        print('keyw2pos_click_data : ',keyw2pos_click_data)
        print('============================')
        print('CPA:', cpa)
        print('# of install:', metric1_install)
        # precision recall
        P_k_list, R_k_list = \
            self.Rank_P_and_R_k(result_keyw_list, ground_true, topk)
        print('P_k_list:', P_k_list)
        print('R_k_list:', R_k_list)
        print('============================')
        print('rankly_record_for_install_cpa : ',rankly_record_for_install_cpa)
        print('============================')
        print('rankly_record_for_click : ',rankly_record_for_click)
        print('============================')
        print('result_keyw_list : ',result_keyw_list)
        print('============================')
        return result_keyw_list

    def TEST_online(self, account_id, result_keyw_list,
                    ground_true, topk):
        overlap_pos = list(set(ground_true) & set(result_keyw_list))
        self._account_id2keyw2performance_(account_id)
        result_performance_list = list()
        keyw2performance = self.account_id2keyw2performance[account_id]
        metric1_spend = 0
        metric1_install = 0
        overlap_pos2data = dict()
        impression_zero = list()
        click_keyw2data = dict()
        for i in range(len(result_keyw_list)):
            if result_keyw_list[i] in keyw2performance:
                performance = keyw2performance[result_keyw_list[i]]
                installs = performance['installs']
                local_spend = performance['local_spend']
                impressions = performance['impressions']
                clicks = performance['clicks']
                metric1_spend += local_spend
                metric1_install += installs
                if installs > 0:
                    overlap_pos2data[result_keyw_list[i]] = \
                        (impressions, clicks, installs, local_spend)
                if clicks > 0:
                    click_keyw2data[result_keyw_list[i]] = \
                        (impressions, clicks, installs, local_spend)
            else:
                impression_zero.append(result_keyw_list[i])
        print('overlap_pos2data:', overlap_pos2data)
        print('---------------------------------')
        print('impression_zero num : ', len(impression_zero))
        print('---------------------------------')
        print('click_keyw2data : ', click_keyw2data)
        print('click_keyw2data num : ', len(click_keyw2data))
        if metric1_install != 0:
            cpa = metric1_spend / metric1_install
        else:
            cpa = 'None'
        print('============================')
        print('CPA:', cpa)
        print('# of install:', metric1_install)
        # precision recall
        P_k_list, R_k_list = \
            self.Rank_P_and_R_k(result_keyw_list, ground_true, topk)
        print('P_k_list:', P_k_list)
        print('R_k_list:', R_k_list)
        print('============================')
        if metric1_install != 0:
            cumulative_install_list, cumulative_cpa_list, index = \
                self.Cumulative_Part(result_keyw_list, keyw2performance)
            print('cumulative #_of_install :', cumulative_install_list)
            print('cumulative CPA :', cumulative_cpa_list)
            print('============================')
            return cumulative_install_list, cumulative_cpa_list, index
        else:
            return 'None', 'None', 'None'

    def Rank_P_and_R_k(self, pred, ground_true, topk):
        P_k_list, R_k_list = list(), list()
        for i in range(len(topk)):
            pos_num = 0
            for j in range(topk[i]):
                if len(pred) > j:
                    if pred[j] in ground_true:
                        pos_num += 1
            precision = pos_num / topk[i]
            recall = pos_num / len(ground_true)
            P_k_list.append(precision)
            R_k_list.append(recall)
        return P_k_list, R_k_list

    def Cumulative_Part(self, result_keyw_list_topK, keyw2performance):
        hit_num_list = list()
        install_num_list = list()
        local_spend_list = list()
        index = list()
        for i in range(len(result_keyw_list_topK)):
            keyw = result_keyw_list_topK[i]
            if keyw in keyw2performance:
                performance = keyw2performance[keyw]
                impressions = performance['impressions']
                installs = performance['installs']
                local_spend = performance['local_spend']
                if installs == 0:
                    hit_num_list.append(0)
                else:
                    hit_num_list.append(1)
                install_num_list.append(installs)
                local_spend_list.append(local_spend)
                index.append(i+1)
            else:
                hit_num_list.append(0)
                install_num_list.append(0)
                local_spend_list.append(0)
        cumulative_hit_list = list()
        cumulative_install_list = list()
        cumulative_spend_list = list()
        for i in range(len(hit_num_list)):
            if i == 0:
                cumulative_hit_list.append(hit_num_list[i])
                cumulative_install_list.append(install_num_list[i])
                cumulative_spend_list.append(local_spend_list[i])
            else:
                cumulative_hit_list.append(hit_num_list[i-1] + hit_num_list[i])
                cumulative_install_list.\
                    append(cumulative_install_list[i-1] + install_num_list[i])
                cumulative_spend_list.\
                    append(cumulative_spend_list[i-1] + local_spend_list[i])
        cumulative_cpa_list = list()
        for i in range(len(cumulative_hit_list)):
            if cumulative_install_list[i] > 0:
                c_cpa = \
                    cumulative_spend_list[i] / cumulative_install_list[i]
                cumulative_cpa_list.append(c_cpa)
            else:
                cumulative_cpa_list.append(0)
        return cumulative_install_list, cumulative_cpa_list, index

    def TEST_feature_level(self, account_id, neighbor2candidate_keyw_rank_list,
                           prediction_result_KEYW, ground_true):
        neighbor = list(neighbor2candidate_keyw_rank_list.keys())
        for i in range(len(neighbor)):
            candidate_keyw_rank_list = \
                neighbor2candidate_keyw_rank_list[neighbor[i]]
            candidate_keyw_rank_list_overlap_with_prediction = list()
            candidate_keyw_rank_list_diff_with_prediction = list()
            for j in range(len(candidate_keyw_rank_list)):
                if candidate_keyw_rank_list[j] in prediction_result_KEYW:
                    candidate_keyw_rank_list_overlap_with_prediction\
                        .append(candidate_keyw_rank_list[j])
                else:
                    candidate_keyw_rank_list_diff_with_prediction\
                        .append(candidate_keyw_rank_list[j])
            index_list = \
                [(k + 1) for k in range(len(candidate_keyw_rank_list))]
            self.TEST(account_id, candidate_keyw_rank_list,
                      ground_true, index_list)

    def Two_model_Plot(self, x_axis_name, y_axis_name,
                       list1=None, list2=None, m1_name=None,
                       m2_name=None, file_name=None, index=None):
        if list1 != 'None' or list2 != 'None':
            if list1 == 'None':
                index = list2
            else:
                index = list1
            x_axis = [(i+1) for i in range(len(index))]
            if list1 != 'None':
                y1_axis = list1
                plt.plot(x_axis, y1_axis, '-o', color='orange')
            if list2 != 'None':
                y2_axis = list2
                plt.plot(x_axis, y2_axis, '-o', color='green')
            ax = plt.gca()
            plt.legend([m1_name, m2_name], loc=4)
            plt.xlabel(x_axis_name)
            plt.ylabel(y_axis_name)
            plt.savefig(file_name + '.png')
            plt.clf()


def _read_online_submit_potential_keyword_(config):
    # config go to here
    online_FG_PLUS_path = config.to('online_FG_PLUS_path')
    online_BERT_path = config.to('online_BERT_path')
    # model_name -> prediction keyword
    model2potential_keyword = dict()
    for used_model_and_show_DA in [('FG_PLUS', True), ('BERT', False)]:
        used_model = used_model_and_show_DA[0]
        show_DA = used_model_and_show_DA[1]
        if used_model == 'FG_PLUS':
            potential_keyword_path = online_FG_PLUS_path
        elif used_model == 'BERT':
            potential_keyword_path = online_BERT_path
        with open(potential_keyword_path, 'r') as f:
            keyword_str = str(f.read())
        keyword_str_split = keyword_str.split(',')
        keyword_str_split_set = list()
        for i in range(len(keyword_str_split)):
            if keyword_str_split[i] not in keyword_str_split_set:
                keyword_str_split_set.append(keyword_str_split[i])
        model2potential_keyword[used_model] = keyword_str_split_set
    return model2potential_keyword


def _evaluation_for_online_(asa_account_id, evaluation, config):
    # config go to here
    install_keyw_list = config.to('install_keyw_list')
    online_submit_save_path_part1 = config.to('online_submit_save_path_part1')
    target_app_name = config.to('target_app_name')
    # setting
    online_evaluation_topN_KEYW_index = [10 * (i + 1) for i in range(10)]
    # read online submit data
    model2potential_keyword = _read_online_submit_potential_keyword_(config)
    FG_PLUS_potential_keyword = model2potential_keyword['FG_PLUS']
    BERT_potential_keyword = model2potential_keyword['BERT']
    if len(FG_PLUS_potential_keyword) != len(BERT_potential_keyword):
        max_len = \
            max([len(FG_PLUS_potential_keyword), len(BERT_potential_keyword)])
        FG_PLUS_potential_keyword += \
            ['tocken' for i in range(max_len - len(FG_PLUS_potential_keyword))]
        BERT_potential_keyword += \
            ['tocken' for i in range(max_len - len(BERT_potential_keyword))]
    # FG-PLUS online evaluation
    c_install_FGP, c_cpa_FGP, index1 = \
        evaluation.TEST_online(asa_account_id,
                               FG_PLUS_potential_keyword,
                               install_keyw_list,
                               online_evaluation_topN_KEYW_index)
    # BERT online evaluation
    c_install_BERT, c_cpa_BERT, index2 = \
        evaluation.TEST_online(asa_account_id, BERT_potential_keyword,
                               install_keyw_list,
                               online_evaluation_topN_KEYW_index)
    file_name = \
        online_submit_save_path_part1 + 'plot' + target_app_name + '-install'
    evaluation.Two_model_Plot('k', 'cpa', list1=c_cpa_FGP, list2=c_cpa_BERT,
                              m1_name='FG-PLUS', m2_name='BERT',
                              file_name=file_name, index=None)
    file_name = \
        online_submit_save_path_part1 + 'plot' + target_app_name + '-install'
    evaluation.Two_model_Plot('k', '#-of-install', list1=c_install_FGP,
                              list2=c_install_BERT, m1_name='FG-PLUS',
                              m2_name='BERT', file_name=file_name,
                              index=None)
    evaluation.TEST(asa_account_id, install_keyw_list, install_keyw_list,
                    online_evaluation_topN_KEYW_index)



class Performance_Plot:
    def __init__(self,account_id,appannie_and_asa,config):
        # go to config 
        self.config = config
        campaign_setting = self.config.to('campaign_setting')
        determine_test_date = self.config.to('determine_test_date')
        # add Evaluation object and then load asa data
        evaluation = Evaluation(appannie_and_asa)
        evaluation._account_id2mode2date2keyw2performance_(account_id=account_id,campaign_setting=campaign_setting)   
        self.date2keyw2performance = evaluation.account_id2mode2date2keyw2performance[account_id]['date']
        self.keyw2performance_for_interval = evaluation.date2keyw2performanceTOkeyw2performance(self.date2keyw2performance,determine_test_date)
        # setting start_date | end_date 
        self.date_list = list(self.date2keyw2performance.keys())
        self.start_date, self.end_date = determine_test_date[0], determine_test_date[1]
        if self.start_date is None:
            self.start_date = min(self.date_list)
        if self.end_date is None:
            self.end_date = max(self.date_list)

    def daily_performance_calculate(self,model_keyword):
        metric_install_overall, metric_spend_overall = 0,0
        online_date_list = list()
        online_daily_install_model, online_daily_spend_model, online_daily_cpa_model = list(), list(), list()
        for date in self.date_list:
            if date <= self.end_date and self.start_date <= date:
                keyw2performance = self.date2keyw2performance[date]
                online_date_list.append(str(date).split('-')[-1])
                metric_install, metric_spend = 0,0
                keyw_list = list(keyw2performance.keys())
                intersection_keyw = list(set(model_keyword) & set(keyw_list))
                for keyw in intersection_keyw:
                    performance = keyw2performance[keyw]
                    local_spend = performance['local_spend']
                    installs = performance['installs']
                    metric_install += installs
                    metric_install_overall += installs
                    metric_spend += local_spend
                    metric_spend_overall += local_spend
                if metric_install != 0:
                    metric_cpa = metric_spend /  metric_install
                else:
                    metric_cpa = 0
                online_daily_install_model.append(metric_install)
                online_daily_spend_model.append(metric_spend)
                online_daily_cpa_model.append(metric_cpa)
        if metric_install_overall != 0:
            overall_cpa = metric_spend_overall / metric_install_overall
        else:
            overall_cpa = 0
        return online_daily_install_model, online_daily_spend_model, online_daily_cpa_model, metric_install_overall,overall_cpa ,online_date_list

    def cumulative_calculate_for_date(self, online_daily_model):
        online_daily_C_model = list()
        for i in range(len(online_daily_model)):
            if i == 0:
                online_daily_C_model.append(online_daily_model[i])
            else:
                online_daily_C_model.append(online_daily_C_model[-1] + online_daily_model[i])
        return online_daily_C_model

    def cumulative_cpa_calculate_for_date(self, online_daily_install_model,online_daily_spend_model):
        online_daily_cpa_model = list()
        for i in range(len(online_daily_spend_model)):
            if online_daily_install_model[i] == 0:
                online_daily_cpa_model.append(0)
            else:
                online_daily_cpa_model.append(online_daily_spend_model[i] / online_daily_install_model[i])
        return online_daily_cpa_model

    def main(self,bert_keyword,bpr_keyword,fg_plus_keyword,ground_true):
        # bert part
        online_daily_install_bert, online_daily_spend_bert, online_daily_cpa_bert, \
        overall_install_bert,overall_cpa_bert ,online_date_list = \
            self.daily_performance_calculate(bert_keyword)
        # fg-plus part
        online_daily_install_fg_plus, online_daily_spend_fg_plus, online_daily_cpa_fg_plus, \
        overall_install_fg_plus,overall_cpa_fg_plus ,online_date_list = \
            self.daily_performance_calculate(fg_plus_keyword)
        # bpr part
        online_daily_install_bpr, online_daily_spend_bpr, online_daily_cpa_bpr, \
        overall_install_bpr,overall_cpa_bpr ,online_date_list = \
            self.daily_performance_calculate(bpr_keyword)
        # upper bound part
        online_daily_install_UB, online_daily_spend_UB, online_daily_cpa_UB, \
        overall_install_UB,overall_cpa_UB ,online_date_list = \
            self.daily_performance_calculate(ground_true)
        # cumulative daily install part
        online_daily_C_install_bert = self.cumulative_calculate_for_date(online_daily_install_bert)
        online_daily_C_install_fg_plus = self.cumulative_calculate_for_date(online_daily_install_fg_plus)
        online_daily_C_install_bpr = self.cumulative_calculate_for_date(online_daily_install_bpr)        
        online_daily_C_install_UB = self.cumulative_calculate_for_date(online_daily_install_UB)
        print('=========================================')
        print('online_daily_C_install_bert : ',online_daily_C_install_bert)
        print('online_daily_C_install_fg_plus : ',online_daily_C_install_fg_plus)
        print('online_daily_C_install_bpr : ',online_daily_C_install_bpr)
        print('online_daily_C_install_UB : ',online_daily_C_install_UB)
        # cumulative daily spend part
        online_daily_C_spend_bert = self.cumulative_calculate_for_date(online_daily_spend_bert)
        online_daily_C_spend_fg_plus = self.cumulative_calculate_for_date(online_daily_spend_fg_plus)
        online_daily_C_spend_bpr = self.cumulative_calculate_for_date(online_daily_spend_bpr)
        online_daily_C_spend_UB = self.cumulative_calculate_for_date(online_daily_spend_UB)
        # cumulative daily cpa part
        online_daily_C_cpa_bert = self.cumulative_cpa_calculate_for_date(online_daily_C_install_bert,online_daily_C_spend_bert)
        online_daily_C_cpa_fg_plus = self.cumulative_cpa_calculate_for_date(online_daily_C_install_fg_plus,online_daily_C_spend_fg_plus)
        online_daily_C_cpa_bpr = self.cumulative_cpa_calculate_for_date(online_daily_C_install_bpr,online_daily_C_spend_bpr)
        online_daily_C_cpa_UB = self.cumulative_cpa_calculate_for_date(online_daily_C_install_UB,online_daily_C_spend_UB)
        print('=========================================')
        print('online_daily_C_cpa_bert : ',online_daily_C_cpa_bert)
        print('online_daily_C_cpa_fg_plus : ',online_daily_C_cpa_fg_plus)
        print('online_daily_C_cpa_bpr : ',online_daily_C_cpa_bpr)
        print('online_daily_C_cpa_UB : ',online_daily_C_cpa_UB)        
        # X-topK / Y - C_cpa-or-C_install
        c_install_bert, c_cpa_bert, c_spend_bert = self.cumulative_calculate_for_topK(bert_keyword,self.keyw2performance_for_interval) 
        c_install_fg_plus, c_cpa_fg_plus, c_spend_fg_plus = self.cumulative_calculate_for_topK(fg_plus_keyword,self.keyw2performance_for_interval)
        c_install_bpr, c_cpa_bpr, c_spend_bpr = self.cumulative_calculate_for_topK(bpr_keyword,self.keyw2performance_for_interval)        
        print('=========================================')
        print('c_install_bert : ',c_install_bert)
        print('c_install_fg_plus : ',c_install_fg_plus)
        print('c_install_bpr : ',c_install_bpr)
        print('=========================================')
        print('c_cpa_bert : ',c_cpa_bert)
        print('c_cpa_fg_plus : ',c_cpa_fg_plus)
        print('c_cpa_bpr : ',c_cpa_bpr)
        # plot
        self.plot(online_daily_install_bert,online_daily_install_fg_plus,online_daily_install_bpr,online_daily_install_UB,\
            online_date_list,'-daily_install',self.config,xlabel='date',ylabel='install')
        self.plot(online_daily_cpa_bert,online_daily_cpa_fg_plus,online_daily_cpa_bpr,online_daily_cpa_UB, \
            online_date_list,'-daily_cpa',self.config,xlabel='date',ylabel='cpi')
        #
        self.plot(online_daily_C_install_bert,online_daily_C_install_fg_plus,online_daily_C_install_bpr,online_daily_C_install_UB,\
            online_date_list,'-daily_C_install',self.config,xlabel='date',ylabel='cumulative install')
        self.plot(online_daily_C_cpa_bert,online_daily_C_cpa_fg_plus,online_daily_C_cpa_bpr,online_daily_C_cpa_UB, \
            online_date_list,'-daily_C_cpa',self.config,xlabel='date',ylabel='cumulative cpi')
        #
        self.plot(c_install_bert,c_install_fg_plus,c_install_bpr,None,  \
            None,'-C_install',self.config,xlabel='topK',ylabel='cumulative install')
        self.plot(c_cpa_bert,c_cpa_fg_plus,c_cpa_bpr,None,  \
            None,'-C_cpa',self.config,xlabel='topK',ylabel='cumulative cpi')


    def plot(self,list0,list1, list2 , list3=None ,date_list=None,file_main_name=None,config=None,xlabel=None,ylabel=None):
        offline_save_where = config.to('offline_save_where')
        path = offline_save_where + file_main_name + '.png'
        x0 = range(0,len(list0))
        x1=range(0,len(list1))
        x2=range(0,len(list2))
        if list3 is not None:
            x3=range(0,len(list3))
        #plt.figure(figsize=(10, 6))
        plt.plot(x0,list0,'-o',label="BERT")
        plt.plot(x1,list1,'-o',label="FG-PLUS")
        plt.plot(x2,list2,'-o',label="BPR")
        if list3 is not None:
            plt.plot(x3,list3,label="Overall")
        plt.legend(loc='lower right')
        if date_list is not None:
            plt.xticks([i for i in range(len(date_list))],date_list,fontsize=5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(x1)
        plt.savefig(path)
        plt.clf()

    def cumulative_calculate_for_topK(self,keyw_list,keyw2performance_for_interval):
        c_install, c_cpa, c_spend = list(), list(), list()
        for keyw in keyw_list: 
            if keyw in keyw2performance_for_interval:
                performance = keyw2performance_for_interval[keyw]
                installs = performance['installs']
                local_spend = performance['local_spend']
                if len(c_install) == 0:
                    c_install.append(installs)
                    c_spend.append(local_spend)
                else:
                    c_install.append(c_install[-1] + installs)
                    c_spend.append(c_spend[-1] + local_spend)
                if len(c_cpa) == 0:
                    if installs > 0 :
                        cpa = local_spend / installs
                    else:
                        cpa = 0
                    c_cpa.append(cpa)
                else:
                    if c_install[-1] > 0 :
                        cpa = c_spend[-1] / c_install[-1]
                    else:
                        cpa = 0
                    c_cpa.append(cpa)
            else:
                if len(c_install) == 0:
                    c_install.append(0)
                    c_cpa.append(0)
                    c_spend.append(0)
                else:
                    c_install.append(c_install[-1])
                    c_cpa.append(c_cpa[-1])
                    c_spend.append(c_spend[-1])    
        return c_install, c_cpa, c_spend

    def _read_history_submit_keyword_(self,path):
        with open(path,'r') as f:
            keyword_str = str(f.read())
        keyword_str_split = keyword_str.split(',')
        keyword_str_split_set = list()
        for i in range(len(keyword_str_split)):
            if keyword_str_split[i] not in keyword_str_split_set:
                keyword_str_split_set.append(keyword_str_split[i])
        return keyword_str_split_set
