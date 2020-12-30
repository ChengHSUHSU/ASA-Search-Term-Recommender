import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pyprind
from util import language_classifier
from util import Cosine_Similarity


class BPR(nn.Module):
    def __init__(self, input_dim, impression_test, candidate_keyw, keyw2embedding):
        super(BPR, self).__init__()
        # train part
        self.tanh = nn.ReLU()
        self.W = nn.Parameter(torch.empty(input_dim, 1))
        self.W = torch.nn.init.xavier_uniform_(self.W)
        # forecast part for impression_test
        if impression_test is not None:
            impression_test_emb = list()
            for keyw in impression_test:
                keyw_emb = keyw2embedding[keyw].reshape(1,-1)
                keyw_emb = torch.tensor(keyw_emb)
                impression_test_emb.append(keyw_emb)
            self.impression_test_emb = torch.cat(impression_test_emb,0)
        # forecast part for candidate_keyw
        candidate_keyw_emb = list()
        for keyw in candidate_keyw:
            keyw_emb = keyw2embedding[keyw].reshape(1,-1)
            keyw_emb = torch.tensor(keyw_emb)
            candidate_keyw_emb.append(keyw_emb)
        self.candidate_keyw_emb = torch.cat(candidate_keyw_emb,0)
        self.candidate_keyw = candidate_keyw
        if impression_test is not None:
            self.impression_test = impression_test

    def forward(self, batch_data=None, bpr_mode=None):
        if bpr_mode == 'train':
            keyw0_embedding, keyw1_embedding = batch_data[0], batch_data[1] # (bz,dim) ,(bz,dim)
            W_keyw1 = torch.matmul(keyw1_embedding, self.W)
            W_keyw_diff = (W_keyw0 - W_keyw1).view(-1)
            log_prob = F.logsigmoid(W_keyw_diff).sum()
            return -log_prob 
        elif bpr_mode == 'pred-impression_test':
            r_impression_test = \
                torch.matmul(self.impression_test_emb, self.W).cpu().detach().numpy().reshape(-1)
            r_and_candidatte_keyw_list = list()
            for i in range(len(self.impression_test)):
                r_and_candidatte_keyw_list.append([r_impression_test[i], self.impression_test[i]])
            r_and_impression_test = sorted(r_and_candidatte_keyw_list,reverse=True)
            return r_and_impression_test
        elif bpr_mode == 'pred-candidate_keyw':
            r_candidate_keyw = \
                torch.matmul(self.candidate_keyw_emb, self.W).cpu().detach().numpy().reshape(-1)
            r_and_candidatte_keyw_list = list()
            for i in range(len(self.candidate_keyw)):
                r_and_candidatte_keyw_list.append([r_candidate_keyw[i], self.candidate_keyw[i]])
            r_and_candidate_keyw_list = sorted(r_and_candidatte_keyw_list,reverse=True)
            return r_and_candidate_keyw_list


class Basic_BPR:
    def __init__(self,config,mode='train'):
        impression_train = config.to('impression_train')
        install_train = config.to('install_train')
        print('install_train: ',install_train)
        impression_test = config.to('impression_test')
        candidate_keyw = config.to('candidate_keyw')
        self.keyw2embedding = config.to('keyw2embedding')
        self.keyw2performance_for_interval = \
            config.to('keyw2performance_for_interval')
        self.keyw2performance_for_determine_date = \
            config.to('keyw2performance_for_determine_date')
        self.mode = mode
        if self.mode == 'train':
            self.impression_test = \
                list((set(impression_test) - set(impression_train)) )
            self.keyw_instance_list = \
                list(self.keyw2performance_for_determine_date.keys())
            self.candidate_keyw = \
                list(set(candidate_keyw) - set(impression_train))
        elif self.mode == 'forecast':
            self.keyw_instance_list = \
                list(self.keyw2performance_for_interval.keys())
            self.candidate_keyw = \
                list(set(candidate_keyw) - set(impression_train) - set(impression_test))            
        # parameter setting
        self.epoch_num = 20
        self.batch_size = 32
        if self.mode == 'train':
            self.bpr_model = BPR(768, self.impression_test, self.candidate_keyw, self.keyw2embedding)
        elif self.mode == 'forecast':
            self.bpr_model = BPR(768, None, self.candidate_keyw, self.keyw2embedding)
        self.optimizer = optim.Adam(self.bpr_model.parameters(), lr=0.001)

        # multi-label part
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # language filter for output keyword
        self.online_submit_keyw_num = config.to('online_submit_keyw_num')

    def _language_filter_for_rank_result_candidate_keyw_(self,
                                                         rank_result_keyw):
        # language -> pos_keyw (install > 0)
        lanuage2pos_keyw = dict()
        keyw_list = list(self.keyw2performance_for_interval.keys())
        for keyw in keyw_list:
            performance = self.keyw2performance_for_interval[keyw]
            installs = performance['installs']
            LN = language_classifier(keyw)
            if installs > 0 and len(LN) == 1:
                if LN[0] not in lanuage2pos_keyw:
                    lanuage2pos_keyw[LN[0]] = list()
                lanuage2pos_keyw[LN[0]].append(keyw)
        # language -> rank_result depend on language
        lanuage2rank_result = dict()
        used_language = list(lanuage2pos_keyw.keys())
        for ln in used_language:
            lanuage2rank_result[ln] = list()
        for keyw in rank_result_keyw:
            LN = language_classifier(keyw)
            if len(LN) == 1:
                if LN[0] in used_language:
                    lanuage2rank_result[LN[0]].append(keyw)
        # language with used_num
        used_num_and_language, pos_keyw_num = list(), 0
        for ln in used_language:
            pos_keyw_list = lanuage2pos_keyw[ln]
            used_num_and_language.append([len(pos_keyw_list), ln])
            pos_keyw_num += len(pos_keyw_list)
        for keyw_num_and_ln in used_num_and_language:
            keyw_num_and_ln[0] = \
                int((keyw_num_and_ln[0] / pos_keyw_num) * self.online_submit_keyw_num)
        used_num_and_language = sorted(used_num_and_language, reverse=True)
        # rank_result_keyw based on Langauge Filter
        rank_result_keyw_based_on_LF = list()
        for used_num_and_ln in used_num_and_language:
            used_num = used_num_and_ln[0]
            ln = used_num_and_ln[1]
            rank_result_keyw = lanuage2rank_result[ln]
            rank_result_keyw_with_used_num = rank_result_keyw[:used_num]
            rank_result_keyw_based_on_LF += rank_result_keyw_with_used_num
        if len(rank_result_keyw_based_on_LF) < self.online_submit_keyw_num:
            max_used_num_and_language = max(used_num_and_language)
            used_num = max_used_num_and_language[0]
            ln = max_used_num_and_language[1]
            residual_used_num = \
                self.online_submit_keyw_num - len(rank_result_keyw_based_on_LF)
            rank_result_keyw = lanuage2rank_result[ln]
            residual_rank_result_keyw = \
            rank_result_keyw[used_num : used_num + residual_used_num]
            rank_result_keyw_based_on_LF += residual_rank_result_keyw
        elif len(rank_result_keyw_based_on_LF) > self.online_submit_keyw_num:
            rank_result_keyw_based_on_LF = \
                rank_result_keyw_based_on_LF[:self.online_submit_keyw_num]
        return rank_result_keyw_based_on_LF

    def _pair_determine_order_based_on_performance_(self, keyw1, keyw2):
        if self.mode == 'train':
            keyw1_performance = \
                self.keyw2performance_for_determine_date[keyw1]
            keyw2_performance = \
                self.keyw2performance_for_determine_date[keyw2]
        elif self.mode == 'forecast':
            keyw1_performance = \
                self.keyw2performance_for_interval[keyw1]
            keyw2_performance = \
                self.keyw2performance_for_interval[keyw2]
        # keyw1 part
        keyw1_impressions = keyw1_performance['impressions']
        keyw1_installs = keyw1_performance['installs']
        keyw1_local_spend = keyw1_performance['local_spend']
        if keyw1_installs > 0:
            keyw1_cpa = keyw1_local_spend / keyw1_installs
        else:
            keyw1_cpa = None
        # keyw2 part
        keyw2_impressions = keyw2_performance['impressions']
        keyw2_installs = keyw2_performance['installs']
        keyw2_local_spend = keyw2_performance['local_spend']
        if keyw2_installs > 0:
            keyw2_cpa = keyw2_local_spend / keyw2_installs
        else:
            keyw2_cpa = None
        # pair compare
        if keyw1_installs > 0 and keyw2_installs > 0:
            if keyw1_cpa < keyw2_cpa:
                return (keyw1, keyw2)
            elif keyw1_cpa > keyw2_cpa:
                return (keyw2, keyw1)
            else:
                return None
        elif keyw1_installs > 0 and keyw2_installs == 0:
            return (keyw1, keyw2)
        elif keyw1_installs == 0 and keyw2_installs > 0:
            return (keyw2, keyw1)
        elif keyw1_installs == 0 and keyw2_installs == 0:
            if keyw1_local_spend < keyw2_local_spend:
                return (keyw1, keyw2)
            elif keyw1_local_spend > keyw2_local_spend:
                return (keyw2, keyw1)
            else:
                return None

    def _pair_ordered_keyw_(self, keyw_instance_list):
        keyw_instance_list = list(set(keyw_instance_list))
        pair_ordered_keyw_list = list()
        for i in pyprind.prog_bar(range(len(keyw_instance_list))):
            keyw1 = keyw_instance_list[i]
            #keyw1_ln = set(language_classifier(keyw1))
            for keyw2 in keyw_instance_list:
                #keyw2_ln = set(language_classifier(keyw2))
                if keyw1 != keyw2 :#and len(keyw1_ln & keyw2_ln) > 0:
                    compared_result = \
                        self._pair_determine_order_based_on_performance_(keyw1,keyw2)
                    if compared_result is not None:
                        pair_ordered_keyw_list.append(compared_result)
        return pair_ordered_keyw_list

    def _batching_function_(self,pair_ordered_keyw_list, batch_size):
        batch_pair_ordered_keyw_list = list()
        batch_num = int(len(pair_ordered_keyw_list) / batch_size) + 1
        for i in range(batch_num):
            batch_data_column_format = \
                pair_ordered_keyw_list[(i * batch_size): (i + 1) * batch_size]
            if len(batch_data_column_format) > 0:
                batch_data_row_format = \
                    self._transform_into_row_with_tensor_(batch_data_column_format)
                batch_pair_ordered_keyw_list.append(batch_data_row_format)
        return batch_pair_ordered_keyw_list

    def _transform_into_row_with_tensor_(self, batch_data_column_format):
        batch_data_row_format = [list(), list()]
        for pair_keyw in batch_data_column_format:
            keyw0, keyw1 = pair_keyw[0], pair_keyw[1]
            keyw_emb0 = self.keyw2embedding[keyw0].reshape(1, -1)
            keyw_emb0 = torch.tensor(keyw_emb0)
            keyw_emb1 = self.keyw2embedding[keyw1].reshape(1, -1)
            keyw_emb1 = torch.tensor(keyw_emb1)
            batch_data_row_format[0].append(keyw_emb0)
            batch_data_row_format[1].append(keyw_emb1)
        batch_data_row_format[0] = \
            torch.cat(batch_data_row_format[0], 0)
        batch_data_row_format[1] = \
            torch.cat(batch_data_row_format[1], 0)
        return batch_data_row_format

    def main(self):
        # pair compared part
        pair_ordered_keyw_list = self._pair_ordered_keyw_(self.keyw_instance_list)
        # train part
        for epoch in range(self.epoch_num):
            batch_pair_ordered_keyw_list = \
                self._batching_function_(pair_ordered_keyw_list, self.batch_size)
            loss = 0
            for i in range(len(batch_pair_ordered_keyw_list)):
                batch_data = batch_pair_ordered_keyw_list[i]
                batch_loss = self.bpr_model(batch_data, bpr_mode='train')
                loss += batch_loss
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            print('avg loss [Regression Part] : ', loss / len(batch_pair_ordered_keyw_list))
        # pred in impression_test
        if self.mode == 'train':
            r_and_impression_test = self.bpr_model(None, bpr_mode='pred-impression_test')
            rank_result_impression_test = list()
            for r_and_impression_test_keyw in r_and_impression_test:
                rank_result_impression_test.append(r_and_impression_test_keyw[1])
        # pred in candidate_keyw
        r_and_candidatte_keyw_list = self.bpr_model(None, bpr_mode='pred-candidate_keyw')
        rank_result_candidate_keyw = list()
        for r_and_candidatte_keyw in r_and_candidatte_keyw_list:
            rank_result_candidate_keyw.append(r_and_candidatte_keyw[1])
        if self.mode == 'train':
            rank_result_candidate_keyw = \
                self._language_filter_for_rank_result_candidate_keyw_(rank_result_candidate_keyw)
            return rank_result_impression_test, rank_result_candidate_keyw
        elif self.mode == 'forecast':
            rank_result_candidate_keyw = \
                self._language_filter_for_rank_result_candidate_keyw_(rank_result_candidate_keyw)
            return rank_result_candidate_keyw


class BERT:
    def __init__(self,config,mode='train'):
        impression_train = config.to('impression_train')
        impression_test = config.to('impression_test')
        candidate_keyw = config.to('candidate_keyw')
        self.target_app_name = config.to('target_app_name')
        self.keyw2embedding = config.to('keyw2embedding')
        self.online_submit_keyw_num = config.to('online_submit_keyw_num')
        self.mode = mode
        if self.mode == 'train':
            self.impression_test = \
                list((set(impression_test) - set(impression_train)))
            self.candidate_keyw = \
                list(set(candidate_keyw) - set(impression_train))
        elif self.mode == 'forecast':
            self.candidate_keyw = \
                list(set(candidate_keyw) - set(impression_train) - set(impression_test))

    def _directly_use_app_embedding_and_CS_for_KEYW_rank_(self):
        # target app embedding
        target_app_embedding = self.keyw2embedding[self.target_app_name]
        if self.mode == 'train':
            # impression test keyword embedding
            impression_test_embedding = \
                [self.keyw2embedding[keyw] for keyw in self.impression_test]       
        # candidate keyword embedding
        candidate_keyw_embedding = \
            [self.keyw2embedding[keyw] for keyw in self.candidate_keyw]
        if self.mode == 'train':
            # impression test prediciton
            csm_list_sorted_TOP = \
                Cosine_Similarity(target_app_embedding,
                                impression_test_embedding,
                                self.impression_test,
                                len(self.impression_test))
            self.rank_result_impression_test = \
                [csm_list_sorted_TOP[i][1] for i in range(len(csm_list_sorted_TOP))]
        # candidate keyword prediciton
        csm_list_sorted_TOP = \
            Cosine_Similarity(target_app_embedding,
                            candidate_keyw_embedding,
                            self.candidate_keyw,
                            len(self.candidate_keyw))
        self.rank_result_candidate_keyw = \
            [csm_list_sorted_TOP[i][1] for i in range(len(csm_list_sorted_TOP))]

    def main(self):
        self._directly_use_app_embedding_and_CS_for_KEYW_rank_()
        if self.mode == 'train':
            return self.rank_result_impression_test, self.rank_result_candidate_keyw       
        elif self.mode == 'forecast':
            return self.rank_result_candidate_keyw[:self.online_submit_keyw_num]


class FG_PLUS:
    def __init__(self,config,mode='train'):
        self.use_rule_organic_as_seed = True
        self.use_pos_keyw_as_seed = False
        self.mode = mode
        self.config = config
        impression_train = config.to('impression_train')
        self.install_train = config.to('install_train')
        self.install_test = config.to('install_test')
        impression_test = config.to('impression_test')
        candidate_keyw = config.to('candidate_keyw')
        self.keyw2embedding = config.to('keyw2embedding')
        self.target_app_name = config.to('target_app_name')
        self.app_name2neighbor = config.to('app_name2neighbor')
        if self.mode == 'train':
            self.impression_test = \
                list((set(impression_test) - set(impression_train)))
            self.candidate_keyw = \
                list(set(candidate_keyw) - set(impression_train))
        elif self.mode == 'forecast':
            self.candidate_keyw = \
                list(set(candidate_keyw) - set(impression_train) - set(impression_test))
        # language filter for output keyword
        self.online_submit_keyw_num = config.to('online_submit_keyw_num')


    def _rank_keyw_for_all_cluster_by_app_embedding_(self,
                                                     pep8_tocken,
                                                     neighbor2used_num, 
                                                     config):
        neighbor2candidate_keyw_rank_list = pep8_tocken
        # config go to here
        target_app_embedding = self.keyw2embedding[self.target_app_name]
        potential_keyword = list()
        neighbor2rank_level_keyw_list = dict()
        neighbor = list(neighbor2candidate_keyw_rank_list.keys())
        for i in range(len(neighbor)):
            neighbor2rank_level_keyw_list[neighbor[i]] = list()
            if neighbor[i] in neighbor2used_num:
                used_num = neighbor2used_num[neighbor[i]]
                candidate_keyw_rank_list = \
                    neighbor2candidate_keyw_rank_list[neighbor[i]]
                epoch = math.ceil(len(candidate_keyw_rank_list) / used_num)
                for j in range(epoch):
                    candidate_keyw_rank_list_j_to_j_plus_1 = \
                        candidate_keyw_rank_list[
                            j * (used_num): (j + 1) * used_num]
                    neighbor2rank_level_keyw_list[neighbor[i]] \
                        .append(candidate_keyw_rank_list_j_to_j_plus_1)
        rank_level2candidate_keyword = dict()
        for i in range(len(neighbor)):
            rank_level_keyw_list = neighbor2rank_level_keyw_list[neighbor[i]]
            for j in range(len(rank_level_keyw_list)):
                if j not in rank_level2candidate_keyword:
                    rank_level2candidate_keyword[j] = list()
                rank_level2candidate_keyword[j] += rank_level_keyw_list[j]
        rank_level = list(rank_level2candidate_keyword.keys())
        for i in range(len(rank_level)):
            candidate_keyword = rank_level2candidate_keyword[i]

            candidate_keyword_embedding = [
                self.keyw2embedding[candidate_keyword[j]]
                for j in range(len(candidate_keyword))]
            csm_list_sorted_TOP = \
                Cosine_Similarity(target_app_embedding,
                                candidate_keyword_embedding,
                                candidate_keyword,
                                TOP_N=len(candidate_keyword))
            ranked_keyword = [
                csm_list_sorted_TOP[j][1]
                for j in range(len(csm_list_sorted_TOP))]
            rank_level2candidate_keyword[i] = ranked_keyword
        return rank_level2candidate_keyword


    def _neighbor2source_keyw_rank_(self,neighbor,
                                       source_keyw, 
                                       config,
                                       save=False):
        # neighbor -> source_keyw_rank_list (non sorted)
        neighbor2source_keyw_rank_list = dict()
        for i in range(len(neighbor)):
            neighbor2source_keyw_rank_list[neighbor[i]] = list()
        neighbor_embedding = \
            [self.keyw2embedding[neighbor[j]] for j in range(len(neighbor))]
        for i in range(len(source_keyw)):
            source_keyword = source_keyw[i]
            source_keyw_embedding = self.keyw2embedding[source_keyword]
            csm_list_sorted_TOP = \
                Cosine_Similarity(source_keyw_embedding,
                                neighbor_embedding,
                                neighbor,
                                TOP_N=len(neighbor))
            max_csm_keyw = csm_list_sorted_TOP[0][1]
            max_csm = csm_list_sorted_TOP[0][0]
            neighbor2source_keyw_rank_list[max_csm_keyw]\
                .append([max_csm, source_keyword])
        # sorted
        threshold_for_source_keyw = 0
        for i in range(len(neighbor)):
            keyw_rank_list = \
                sorted(neighbor2source_keyw_rank_list[neighbor[i]],
                    reverse=True)
            source_keyw_rank_list = list()
            for j in range(len(keyw_rank_list)):
                if isinstance(float(keyw_rank_list[j][0]), float):
                    source_keyw_rank_list.append(keyw_rank_list[j][1])
                else:
                    print(keyw_rank_list[j][0])
                    print(type(keyw_rank_list[j][0]))
            neighbor2source_keyw_rank_list[neighbor[i]] = \
                source_keyw_rank_list
        # save neighbor2source_keyw_rank_list
        online_submit_save_path_part1 = None
        date = None
        if save:
            today = str(date.today())
            path = \
                online_submit_save_path_part1 + \
                'history/' + \
                self.target_app_name + \
                '-' + today + '.txt'
            try:
                os.remove(path)
            except FileNotFoundError:
                a = 0
            with open(path, 'w') as f:
                for i in range(len(neighbor)):
                    source_keyw_rank_list = \
                        neighbor2source_keyw_rank_list[neighbor[i]]
                    f.write(neighbor[i] + ' => ')
                    for j in range(len(source_keyw_rank_list)):
                        f.write(source_keyw_rank_list[j])
                        f.write(',')
                    if len(neighbor) - 1 != i:
                        f.write('\n')
        return neighbor2source_keyw_rank_list


    def _neighbor_extract_potential_keyw_(self, neighbor,
                                          source_keyw,
                                          refer_keyw_language,
                                          neighbor2used_num,
                                          config):
        target_neighbor2source_keyw_rank_list = \
            self._neighbor2source_keyw_rank_(neighbor,
                                             source_keyw,
                                             config,
                                             save=False)
        # rank level -> potential keyword
        pep8_tocken = target_neighbor2source_keyw_rank_list
        rank_level2candidate_keyword = \
            self._rank_keyw_for_all_cluster_by_app_embedding_(pep8_tocken,
                                                              neighbor2used_num,
                                                              config)
        # filter LN by used_LN
        rank_result_list = list()
        rank_level = list(rank_level2candidate_keyword.keys())
        for i in range(len(rank_level)):
            candidate_keyword = rank_level2candidate_keyword[i]
            rank_result_list += candidate_keyword
        rank_result_expect = list()
        for i in range(len(rank_result_list)):
            text = rank_result_list[i]
            used_language = language_classifier(text)
            if len(set(used_language) & set(refer_keyw_language)) > 0:
                rank_result_expect.append(text)
        return rank_result_expect


    def main(self):
        # determine used_LN
        self.used_language = list()
        if self.mode == 'train':
            for keyw in list(self.install_train):
                self.used_language += language_classifier(keyw)
        elif self.mode == 'forecast':
            for keyw in list(self.install_train + self.install_test):
                self.used_language += language_classifier(keyw)
        self.used_language = list(set(self.used_language))
        # determine seed (neighbor)
        if self.use_rule_organic_as_seed:
            neighbor = self.app_name2neighbor[self.target_app_name]
        if self.use_pos_keyw_as_seed:
            if self.mode == 'train':
                neighbor = self.install_train
            elif self.mode == 'forecast':
                neighbor = list(set(self.install_train + self.install_test))
        # determine neighbor2num
        neighbor2used_num = dict()
        for i in range(len(neighbor)):
            neighbor2used_num[neighbor[i]] = 1
        # selected feature -> potential
        if self.mode == 'train':
            source_keyw = self.impression_test
            rank_result_impression_test = self._neighbor_extract_potential_keyw_(neighbor,
                                                                                 source_keyw,
                                                                                 self.used_language,
                                                                                 neighbor2used_num,
                                                                                 self.config)
            rank_result_candidate_keyw = self._neighbor_extract_potential_keyw_(neighbor,
                                                                                self.candidate_keyw,
                                                                                self.used_language,
                                                                                neighbor2used_num,
                                                                                self.config)
            return rank_result_impression_test, rank_result_candidate_keyw[:self.online_submit_keyw_num]
        elif self.mode == 'forecast':
            rank_result_candidate_keyw = self._neighbor_extract_potential_keyw_(neighbor,
                                                                                self.candidate_keyw,
                                                                                self.used_language,
                                                                                neighbor2used_num,
                                                                                self.config)
            return rank_result_candidate_keyw[:self.online_submit_keyw_num]
