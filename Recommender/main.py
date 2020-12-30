import argparse
from config import Parameter_and_Processed_Data
from app_organic_recall_from_appannie import _app_organic_recall_from_appannie_
from online_data_recall_from_asa import _online_data_recall_from_asa_
from embedding import _initial_embedding_by_BERT_
from model import Basic_BPR, BERT, FG_PLUS
from evaluation import Evaluation
from util import forecasting_result_TO_endpoint
from crawl_from_db import PULL_DATA_from_DB



class ASA_Supervised_Serving:
    def __init__(self,config_args):
        self.appannie_and_asa = PULL_DATA_from_DB().appannie_and_asa
        self.config = self._to_self_def_config_(config_args)


    def _to_self_def_config_(self,config_args):
        # go to config
        config = Parameter_and_Processed_Data()
        config.to_config(data=config_args.target_app_id,
                        data_name='target_app_id',
                        come_from='initial_setting')
        config.to_config(data=config_args.asa_account_id,
                        data_name='asa_account_id',
                        come_from='initial_setting')
        config.to_config(data=config_args.top100_filename,
                        data_name='top100_filename',
                        come_from='initial_setting')
        config.to_config(data=config_args.rank_of_target_app,
                        data_name='rank_of_target_app',
                        come_from='initial_setting')
        config.to_config(data=config_args.rank_of_top100_app,
                        data_name='rank_of_top100_app',
                        come_from='initial_setting')
        config.to_config(data=config_args.rank_of_related_app,
                        data_name='rank_of_related_app',
                        come_from='initial_setting')
        config.to_config(data=config_args.online_submit_keyw_num,
                        data_name='online_submit_keyw_num',
                        come_from='initial_setting')
        config.to_config(data=config_args.determine_train_date,
                        data_name='determine_train_date',
                        come_from='initial_setting')
        config.to_config(data=config_args.determine_test_date,
                        data_name='determine_test_date',
                        come_from='initial_setting')
        config.to_config(data=config_args.determine_forecast_date,
                        data_name='determine_forecast_date',
                        come_from='initial_setting')
        if len(config_args.campaign_setting) == 0:
            config_args.campaign_setting = None
        config.to_config(data=config_args.campaign_setting,
                        data_name='campaign_setting',
                        come_from='initial_setting')
        config.to_config(data=config_args.BPR_endpoint_path,
                        data_name='BPR_endpoint_path',
                        come_from='initial_setting')
        return config        


    def main(self, mode='forecast'):
        # Data Processing
        self.config = _app_organic_recall_from_appannie_(self.appannie_and_asa, self.config)
        self.config = _online_data_recall_from_asa_(show_DA_for_Data=True,config=self.config)
        self.config = _initial_embedding_by_BERT_(initial=True,new_word=None,config=self.config)
        if mode == 'train':
            # Modeling
            bpr_offline_rank_result, bpr_without_asa_rank_result = Basic_BPR(self.config,mode='train').main()
            bert_offline_rank_result, bert_without_asa_rank_result = BERT(self.config,mode='train').main()
            fg_plus_offline_rank_result, fg_plus_without_asa_rank_result = FG_PLUS(self.config,mode='train').main()
            # Evaluation
            evaluation = Evaluation(self.appannie_and_asa)
            ground_truth = self.config.to('install_test_in_appannie')
            asa_account_id = self.config.to('asa_account_id')
            print('---------------' + 'BPR' + '---------------')
            evaluation.TEST(asa_account_id, bpr_offline_rank_result,ground_truth,self.config)
            print('---------------' + 'BERT' + '---------------')
            evaluation.TEST(asa_account_id, bert_offline_rank_result,ground_truth,self.config)
            print('---------------' + 'FG-PLUS' + '---------------')
            evaluation.TEST(asa_account_id, fg_plus_offline_rank_result,ground_truth,self.config)
            print('---------------Upper_Bound---------------')
            evaluation.TEST(asa_account_id, ground_truth,ground_truth,self.config)

        elif mode == 'forecast':
            # Modeling
            bpr_online_submit = Basic_BPR(self.config,mode='forecast').main()
            # Online Submit
            BPR_endpoint_path = self.config.to('BPR_endpoint_path')
            forecasting_result_TO_endpoint(forecasting_result=bpr_online_submit,
                                           path=BPR_endpoint_path)


if __name__ == '__main__':
    # argument setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_app_id', type=int, default=584787808)
    parser.add_argument('--asa_account_id', type=int, default=2083150)
    parser.add_argument('--campaign_setting', type=list, default=list())
    parser.add_argument('--top100_filename', type=str, default='top100_photo_tw_dataset.txt')
    parser.add_argument('--determine_train_date', type=list, default=[None,'2020-11-18'])
    parser.add_argument('--determine_test_date', type=list, default=['2020-11-19','2020-11-26'])
    parser.add_argument('--determine_forecast_date', type=list, default=[None,None])
    parser.add_argument('--online_submit_keyw_num', type=int, default=100)
    parser.add_argument('--BPR_endpoint_path', type=str, default='sql-dataset/bpr_online.txt')
    parser.add_argument('--rank_of_target_app', type=int, default=5)
    parser.add_argument('--rank_of_top100_app', type=int, default=5)
    parser.add_argument('--rank_of_related_app', type=int, default=5)
    config_args = parser.parse_args()
    # forward modeling
    asa_supervised_serving = ASA_Supervised_Serving(config_args)
    asa_supervised_serving.main(mode='forecast')
