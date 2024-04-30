from models import TransE, TransH, TransD, TransR
from train import train_trans
from config import CONFIG
from visualizer import visualize_data_list

if __name__ == '__main__':
    config = CONFIG(0)
    # 创建并训练TransE模型
    if config.model_type == 'TransE':
        print("Training TransE model...")
        model = TransE(config.num_entities, config.num_relations, config.embedding_dim).to(config.DEVICE)
    elif config.model_type == 'TransH':
        print("Training TransH model...")
        model = TransH(config.num_entities, config.num_relations, config.embedding_dim).to(config.DEVICE)
    elif config.model_type == 'TransD':
        model = TransD(config.num_entities, config.num_relations, config.embedding_dim).to(config.DEVICE)
    elif config.model_type == 'TransR':
        model = TransR(config.num_entities, config.num_relations, config.embedding_dim).to(config.DEVICE)
    loss_list, acc_list = train_trans(model, config)
    visualize_data_list(loss_list, 'Loss', config.get_prefix, smooth_rate=1)
    visualize_data_list(acc_list, 'Accuracy', config.get_prefix, smooth_rate=1)