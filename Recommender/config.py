import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Parameter_and_Processed_Data:
    def __init__(self):
        self.data_name2data = dict()
        self.history_data_name_and_come_from = list()

    def to_config(self, data, data_name, come_from=None):
        self.data_name2data[data_name] = [data, come_from]
        self.history_data_name_and_come_from.append([data_name,come_from])

    def to(self, data_name, ask_come_from=False):
        if data_name in self.data_name2data:
            if ask_come_from is True:
                print(self.data_name2data[data_name][1])
            return self.data_name2data[data_name][0]
        else:
            print('[ERROR] : data_name not in the config.')
            print('[TRACK] : ' + data_name)

    def print_history(self):
        for data_name_and_come_from in self.history_data_name_and_come_from:
            print('data_name: ',data_name_and_come_from[0])
            print('come_from: ',data_name_and_come_from[1])
            print('-------------------------------------')
