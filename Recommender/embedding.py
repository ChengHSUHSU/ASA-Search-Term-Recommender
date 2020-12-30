import numpy as np
import pyprind
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from transformers import AutoTokenizer, AutoModel


def _initial_embedding_by_BERT_(initial=True, new_word=None, config=None):
    universal_set = config.to('universal_set')
    warnings.simplefilter(action='ignore', category=FutureWarning)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    if initial is True:
        corpus_list = list(set(universal_set))
        keyw2embedding = dict()
        for i in pyprind.prog_bar(range(len(corpus_list))):
            word_tocken = tokenizer(corpus_list[i], return_tensors="pt")
            word_vector = \
                bert_model(**word_tocken)[1].cpu().detach().numpy().reshape(-1)
            keyw2embedding[corpus_list[i]] = word_vector
        function_name = '_initial_embedding_by_BERT_ + initial=True'
        config.to_config(data=keyw2embedding,
                         data_name='keyw2embedding',
                         come_from=function_name)
    else:
        keyw2embedding = config.to('keyw2embedding')
        corpus_list = list(set(new_word) - set(keyw2embedding.keys()))
        if len(corpus_list) > 0:
            for i in pyprind.prog_bar(range(len(corpus_list))):
                word_tocken = tokenizer(corpus_list[i], return_tensors="pt")
                word_vector = bert_model(**word_tocken)[1].cpu().detach()
                keyw2embedding[corpus_list[i]] = \
                    word_vector.numpy().reshape(-1)
        function_name = '_initial_embedding_by_BERT_ + initial=False'
        config.to_config(data=keyw2embedding,
                         data_name='keyw2embedding',
                         come_from=function_name)
    return config


def _FILTER_Graph_Module_(used_model='app-name', config=None):
    # config go to here
    keyw2embedding = config.to('keyw2embedding')
    target_app_name = config.to('target_app_name')
    other_app_name = config.to('other_app_name')
    app_name2neighbor = config.to('app_name2neighbor')
    # app name -> app-embedding
    app_name2embedding = dict()
    if used_model == 'BERT':
        # target part
        target_app_embedding = keyw2embedding[target_app_name]
        app_name2embedding[target_app_name] = target_app_embedding
        # other app part
        other_app_name = list(set(other_app_name))
        for i in range(len(other_app_name)):
            app_name2embedding[other_app_name[i]] = \
                keyw2embedding[other_app_name[i]]
        function_name = '_FILTER_Graph_Module_ + used_model=BERT'
    elif used_model == 'FG_PLUS':
        # target part
        neighbor = app_name2neighbor[target_app_name]
        target_app_embedding = \
            [keyw2embedding[neighbor[i]] for i in range(len(neighbor))]
        target_app_embedding = \
            np.sum(target_app_embedding, 0) / len(target_app_embedding)
        app_name2embedding[target_app_name] = target_app_embedding
        # other app part
        for i in range(len(other_app_name)):
            neighbor = app_name2neighbor[other_app_name[i]]
            neighbor_embedding = \
                [keyw2embedding[neighbor[i]] for i in range(len(neighbor))]
            app_embedding = \
                np.sum(neighbor_embedding, 0) / len(neighbor_embedding)
            app_name2embedding[other_app_name[i]] = app_embedding
        function_name = '_FILTER_Graph_Module_ + used_model=FG_PLUS'
    config.to_config(data=app_name2embedding,
                     data_name='app_name2embedding',
                     come_from=function_name)
    return config
