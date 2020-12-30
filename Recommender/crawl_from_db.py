import pymysql
import pickle


class _PULL_DATA_from_DB_:
    def __init__(self):
        host = "ai-deep-funnel.czyv6iwgcsxw.us-west-2.rds.amazonaws.com"
        db_settings = {"host": host,
                       "port": 3306,
                       "user": "sys_ai_intern",
                       "password": "n9hN3NjR2Wr2JWbS",
                       "db": "jarvis_dev"}
        conn = pymysql.connect(**db_settings)
        asa_daily_performance_feature = ['campaign_name', 'keyword',
                                         'impressions', 'clicks',
                                         'installs', 'local_spend',
                                         'date', 'crawled_time',
                                         'account_id']
        appannie_app_meta_feature = ['product_id', 'product_name',
                                     'main_category', 'other_category_paths',
                                     'description']
        appannie_keyword_list_feature = ['product_id', 'keyword_list1']
        appannie_related_apps_feature = ['src_product_id', 'dst_product_id']
        appannie_keyword_performance_feature = ['product_id', 'keyword',
                                                'search_volume', 'country',
                                                'difficulty', 'rank',
                                                'traffic_share', 'score']
        asa_hourly_record_feature = ['date', 'keyword_text', 'bid_price',
                                     'spend', 'impressions', 'clicks',
                                     'installs']
        dataset_file_dict = {'asa_daily_performance':
                             asa_daily_performance_feature,
                             'appannie_app_meta':
                             appannie_app_meta_feature,
                             'appannie_keyword_list':
                             appannie_keyword_list_feature,
                             'appannie_related_apps':
                             appannie_related_apps_feature,
                             'appannie_keyword_performance':
                             appannie_keyword_performance_feature,
                             'asa_hourly_record':
                             asa_hourly_record_feature}
        dataset_file_list = list(dataset_file_dict.keys())
        dataset_file2dataframe = dict()
        for dataset_file in dataset_file_list:
            with conn.cursor() as cursor:
                column_name = dataset_file_dict[dataset_file]
                if dataset_file not in dataset_file2dataframe:
                    dataset_file2dataframe[dataset_file] = dict()
                for i in range(len(column_name)):
                    command = \
                        'select ' + column_name[i] + ' from ' + dataset_file
                    cursor.execute(command)
                    result = cursor.fetchall()
                    result_list = [result[j][0] for j in range(len(result))]
                    dataset_file2dataframe[dataset_file][column_name[i]] = result_list
        a_dict = {'appannie_and_asa': dataset_file2dataframe}
        file = open('sql-dataset/appannie_and_asa_data.pickle', 'wb')
        pickle.dump(a_dict, file)
        file.close()

class PULL_DATA_from_DB:
    def __init__(self):
        # DATA LOAD
        pickle_path = 'sql-dataset/' + 'appannie_and_asa_data.pickle'
        with open(pickle_path, 'rb') as file:
            a_dict1 = pickle.load(file)
        self.appannie_and_asa = a_dict1['appannie_and_asa']


if __name__ == '__main__':
    pull_data_from_db = PULL_DATA_from_DB()
