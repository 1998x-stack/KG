import json, torch
with open('FB15k/entity_dict.json', 'r') as f:
    entity_dict = json.load(f)

with open('FB15k/relation_dict.json', 'r') as f:
    relation_dict = json.load(f)
    
class CONFIG:
    def __init__(self, device_id=0) -> None:
        self.num_entities = len(entity_dict)
        self.num_relations = len(relation_dict)
        self.batch_size = 128
        self.embedding_dim = 50
        self.train_file_path = 'FB15k/train.txt'
        self.test_file_path = 'FB15k/test.txt'
        self.valid_file_path = 'FB15k/valid.txt'
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.margin = 1.0
        self.model_type = 'TransE'
        self.DEVICE = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    @property
    def get_prefix(self):
        return f'{self.model_type}_emb{self.embedding_dim}_lr{self.learning_rate}_margin{self.margin}'