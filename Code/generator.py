from keras.utils import Sequence
import numpy as np

class get_hir_train_generator(Sequence):
    def __init__(self,news_title,news_entity_index,one_hop_entity,entity_embedding, clicked_news,user_id, news_id, label, batch_size):
        self.title = news_title
        
        self.clicked_news = clicked_news
        self.news_entity_index = news_entity_index
        self.one_hop_entity = one_hop_entity 
        self.entity_embedding = entity_embedding
        
        self.user_id = user_id
        self.doc_id = news_id
        self.label = label
        
        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
    
    def __get_news(self,docids):
        title = self.title[docids]
        entity_ids = self.news_entity_index[docids]
        entity_embedding = self.entity_embedding[entity_ids]
        
        one_hop_ids = self.one_hop_entity[docids]
        one_hop_embedding = self.entity_embedding[one_hop_ids]
        
        return title,entity_embedding, one_hop_embedding
        

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        
        doc_ids = self.doc_id[start:ed]
        title,entity_embedding, one_hop_embedding = self.__get_news(doc_ids)
        
        user_ids = self.user_id[start:ed]
        clicked_ids = self.clicked_news[user_ids]
        
        user_title, user_entity_embedding, user_one_hop = self.__get_news(clicked_ids)
        
        label = self.label[start:ed]
                
        return ([title,entity_embedding, one_hop_embedding, user_title, user_entity_embedding, user_one_hop],[label])

class get_test_generator(Sequence):
    def __init__(self,docids,userids, news_title,news_entity_index,one_hop_entity,entity_embedding, clicked_news,batch_size):
        self.docids = docids
        self.userids = userids
        
        self.title = news_title
        self.clicked_news = clicked_news
        self.news_entity_index = news_entity_index
        self.one_hop_entity = one_hop_entity 

        self.entity_embedding = entity_embedding

        self.batch_size = batch_size
        self.ImpNum = self.docids.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self,docids):
        title = self.title[docids]
        entity_ids = self.news_entity_index[docids]
        entity_embedding = self.entity_embedding[entity_ids]
        
        one_hop_ids = self.one_hop_entity[docids]
        one_hop_embedding = self.entity_embedding[one_hop_ids]
        
        return title,entity_embedding, one_hop_embedding
            
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
            
        docids = self.docids[start:ed]
        
        userisd = self.userids[start:ed]
        clicked_ids = self.clicked_news[userisd]

        title,entity_embedding, one_hop_embedding = self.__get_news(docids)
        user_title, user_entity_embedding, user_one_hop = self.__get_news(clicked_ids)

        return [title,entity_embedding, one_hop_embedding, user_title, user_entity_embedding, user_one_hop]

