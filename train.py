import torch
from torch import optim
from torch.utils.data import DataLoader
from data_loader import KGDataset
from losses import TransLoss
import numpy as np
from tqdm import trange
from tensorboardX import SummaryWriter

writer = SummaryWriter()

# 训练TransE模型
def train_trans(model, config):
    dataset = KGDataset(config.train_file_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # criterion = nn.MarginRankingLoss(margin=1.0)
    criterion = TransLoss(margin=config.margin)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_list = []
    accuracy_list = []

    for epoch in trange(config.num_epochs):
        
        loss_tmp_list = []
        for heads, relations, tails in dataloader:
            heads = heads.to(config.DEVICE)
            relations = relations.to(config.DEVICE)
            tails = tails.to(config.DEVICE)
            
            optimizer.zero_grad()
            pos_scores = model(heads, relations, tails)
            neg_entities = torch.randint(0, model.num_entities, (len(heads),1)).to(config.DEVICE)
            neg_scores = model(heads, relations, neg_entities)
            target = torch.full((len(heads), 1), -1.0).to(config.DEVICE)  # 负样本得分为-1.0
            # loss = criterion(pos_scores, neg_scores, target)
            loss = criterion(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            loss_tmp_list.append(loss.item())
        loss_list.append(np.mean(loss_tmp_list))
        accuracy = test_trans(model, config)
        accuracy_list.append(accuracy)
        
        writer.add_scalar('Loss/train', np.mean(loss_tmp_list), epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Accuracy: {accuracy}")
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {np.mean(loss_tmp_list)}")
    
    return loss_list, accuracy_list

# 测试TransE模型
def test_trans(model, config):
    correct = 0
    total = 0
    dataset = KGDataset(config.test_file_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    with torch.no_grad():
        for heads, relations, tails in dataloader:
            heads = heads.to(config.DEVICE)
            relations = relations.to(config.DEVICE)
            tails = tails.to(config.DEVICE)

            pos_scores = model(heads, relations, tails)
            
            neg_entities = torch.randint(0, model.num_entities, (len(heads), 1)).to(config.DEVICE)
            neg_scores = model(heads, relations, neg_entities)

            for pos_score, neg_score in zip(pos_scores.ravel(), neg_scores.ravel()):
                if pos_score + config.margin < neg_score:
                    correct += 1
                total += 1
            
            # TODO: use predict or batch_prediction to evaluate
            # model.predict(heads, relations, None) == tails

    accuracy = correct / total
    print(f"Accuracy: {accuracy}")
    return accuracy