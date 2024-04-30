import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from config import relation_dict, entity_dict

# 定义数据集类
class KGDataset(Dataset):
    def __init__(self, file_path):
        self.triples = pd.read_csv(file_path, sep='\t', header=None).values.tolist()

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        head = entity_dict[head]
        relation = relation_dict[relation]
        tail = entity_dict[tail]
        return torch.tensor([head]), torch.tensor([relation]), torch.tensor([tail])

# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
if __name__ == '__main__':
    dataset = KGDataset('FB15k/train.txt')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch)
        break