import torch
from torch import nn
from typing import Optional, List

# https://zhuanlan.zhihu.com/p/147542008
# https://github.com/LYuhang/Trans-Implementation.git

# 定义TransE模型
class TransE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # 初始化实体和关系的嵌入向量
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # 初始化嵌入向量
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        # 获取实体和关系的嵌入向量
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)

        # 计算Trans模型中的损失函数
        scores = torch.norm(head_embeddings + relation_embeddings - tail_embeddings, p=2, dim=-1)
        return scores

    def predict(self, head_id: Optional[int] = None, tail_id: Optional[int] = None, relation_id: Optional[int] = None) -> Optional[int]:
        """
        预测函数，根据给定的头实体、尾实体或关系预测另一实体或关系

        Args:
        - head_id (Optional[int]): 头实体的索引
        - tail_id (Optional[int]): 尾实体的索引
        - relation_id (Optional[int]): 关系的索引

        Returns:
        - predicted_entity_or_relation (Optional[int]): 预测的实体或关系索引
        """
        if head_id is not None and tail_id is not None and relation_id is None:
            # 预测关系
            head_embedding = self.entity_embeddings(torch.tensor(head_id))
            tail_embedding = self.entity_embeddings(torch.tensor(tail_id))
            scores = torch.norm(head_embedding + self.relation_embeddings.weight.data - tail_embedding, p=2, dim=-1)
            predicted_relation = torch.argmin(scores).item()
            return predicted_relation
        elif head_id is not None and tail_id is None and relation_id is not None:
            # 预测尾实体
            head_embedding = self.entity_embeddings(torch.tensor(head_id))
            relation_embedding = self.relation_embeddings(torch.tensor(relation_id))
            scores = torch.norm(head_embedding + relation_embedding - self.entity_embeddings.weight.data, p=2, dim=-1)
            predicted_tail = torch.argmin(scores).item()
            return predicted_tail
        elif head_id is None and tail_id is not None and relation_id is not None:
            # 预测头实体
            tail_embedding = self.entity_embeddings(torch.tensor(tail_id))
            relation_embedding = self.relation_embeddings(torch.tensor(relation_id))
            scores = torch.norm(self.entity_embeddings.weight.data + relation_embedding - tail_embedding, p=2, dim=-1)
            predicted_head = torch.argmin(scores).item()
            return predicted_head
        else:
            raise ValueError("Invalid input. Provide either head_id and tail_id to predict relation, head_id and relation_id to predict tail, or tail_id and relation_id to predict head.")

    def batch_predict(self, head_ids: Optional[List[int]] = None, tail_ids: Optional[List[int]] = None, relation_ids: Optional[List[int]] = None) -> Optional[List[int]]:
        """
        批量预测函数，根据给定的头实体、尾实体或关系预测另一实体或关系

        Args:
        - head_ids (Optional[List[int]]): 头实体的索引列表
        - tail_ids (Optional[List[int]]): 尾实体的索引列表
        - relation_ids (Optional[List[int]]): 关系的索引列表

        Returns:
        - predicted_entities_or_relations (Optional[List[int]]): 预测的实体或关系索引列表
        """
        if head_ids is not None and tail_ids is not None and relation_ids is None:
            # 预测关系
            head_embeddings = self.entity_embeddings(torch.LongTensor(head_ids))
            tail_embeddings = self.entity_embeddings(torch.LongTensor(tail_ids))
            scores = torch.norm(head_embeddings[:, None, :] + self.relation_embeddings.weight.data - tail_embeddings[None, :, :], p=2, dim=-1)
            predicted_relations = torch.argmin(scores, dim=1).tolist()
            return predicted_relations
        elif head_ids is not None and tail_ids is None and relation_ids is not None:
            # 预测尾实体
            head_embeddings = self.entity_embeddings(torch.LongTensor(head_ids))
            relation_embeddings = self.relation_embeddings(torch.LongTensor(relation_ids))
            scores = torch.norm(head_embeddings[:, None, :] + relation_embeddings[None, :, :] - self.entity_embeddings.weight.data, p=2, dim=-1)
            predicted_tails = torch.argmin(scores, dim=1).tolist()
            return predicted_tails
        elif head_ids is None and tail_ids is not None and relation_ids is not None:
            # 预测头实体
            tail_embeddings = self.entity_embeddings(torch.LongTensor(tail_ids))
            relation_embeddings = self.relation_embeddings(torch.LongTensor(relation_ids))
            scores = torch.norm(self.entity_embeddings.weight.data[None, :, :] + relation_embeddings[:, None, :] - tail_embeddings[:, None, :], p=2, dim=-1)
            predicted_heads = torch.argmin(scores, dim=1).tolist()
            return predicted_heads
        else:
            raise ValueError("Invalid input. Provide either head_ids and tail_ids to predict relation, head_ids and relation_ids to predict tail, or tail_ids and relation_ids to predict head.")
        

# 定义TransH模型
class TransH(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super(TransH, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # 初始化实体和关系的嵌入向量
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.normal_vectors = nn.Embedding(num_relations, embedding_dim)

        # 初始化嵌入向量
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        nn.init.xavier_uniform_(self.normal_vectors.weight.data)

    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        # 获取实体和关系的嵌入向量
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        
        normal_vector = self.normal_vectors(relations)
        normal_vector = normal_vector / torch.norm(normal_vector, p=2, dim=-1, keepdim=True)
        
        head_proj = head_embeddings - torch.sum(head_embeddings * normal_vector, dim=-1, keepdim=True) * normal_vector
        tail_proj = tail_embeddings - torch.sum(tail_embeddings * normal_vector, dim=-1, keepdim=True) * normal_vector

        # 计算Trans模型中的损失函数
        scores = torch.norm(head_proj + relation_embeddings - tail_proj, p=2, dim=-1)
        return scores

    def predict(self, head_id: Optional[int] = None, tail_id: Optional[int] = None, relation_id: Optional[int] = None) -> Optional[int]:
        """
        预测函数，根据给定的头实体、尾实体或关系预测另一实体或关系

        Args:
        - head_id (Optional[int]): 头实体的索引
        - tail_id (Optional[int]): 尾实体的索引
        - relation_id (Optional[int]): 关系的索引

        Returns:
        - predicted_entity_or_relation (Optional[int]): 预测的实体或关系索引
        """
        if head_id is not None and tail_id is not None and relation_id is None:
            # 预测关系
            head_embedding = self.entity_embeddings(torch.tensor(head_id))
            tail_embedding = self.entity_embeddings(torch.tensor(tail_id))
            scores = torch.norm(head_embedding + self.relation_embeddings.weight.data - tail_embedding, p=2, dim=-1)
            predicted_relation = torch.argmin(scores).item()
            return predicted_relation
        elif head_id is not None and tail_id is None and relation_id is not None:
            # 预测尾实体
            head_embedding = self.entity_embeddings(torch.tensor(head_id))
            relation_embedding = self.relation_embeddings(torch.tensor(relation_id))
            scores = torch.norm(head_embedding + relation_embedding - self.entity_embeddings.weight.data, p=2, dim=-1)
            predicted_tail = torch.argmin(scores).item()
            return predicted_tail
        elif head_id is None and tail_id is not None and relation_id is not None:
            # 预测头实体
            tail_embedding = self.entity_embeddings(torch.tensor(tail_id))
            relation_embedding = self.relation_embeddings(torch.tensor(relation_id))
            scores = torch.norm(self.entity_embeddings.weight.data + relation_embedding - tail_embedding, p=2, dim=-1)
            predicted_head = torch.argmin(scores).item()
            return predicted_head
        else:
            raise ValueError("Invalid input. Provide either head_id and tail_id to predict relation, head_id and relation_id to predict tail, or tail_id and relation_id to predict head.")
    
    def batch_predict(self, head_ids: Optional[List[int]] = None, tail_ids: Optional[List[int]] = None, relation_ids: Optional[List[int]] = None) -> Optional[List[int]]:
        """
        批量预测函数，根据给定的头实体、尾实体或关系预测另一实体或关系

        Args:
        - head_ids (Optional[List[int]]): 头实体的索引列表
        - tail_ids (Optional[List[int]]): 尾实体的索引列表
        - relation_ids (Optional[List[int]]): 关系的索引列表

        Returns:
        - predicted_entities_or_relations (Optional[List[int]]): 预测的实体或关系索引列表
        """
        if head_ids is not None and tail_ids is not None and relation_ids is None:
            # 预测关系
            head_embeddings = self.entity_embeddings(torch.LongTensor(head_ids))
            tail_embeddings = self.entity_embeddings(torch.LongTensor(tail_ids))
            scores = torch.norm(head_embeddings[:, None, :] + self.relation_embeddings.weight.data - tail_embeddings[None, :, :], p=2, dim=-1)
            predicted_relations = torch.argmin(scores, dim=1).tolist()
            return predicted_relations
        elif head_ids is not None and tail_ids is None and relation_ids is not None:
            # 预测尾实体
            head_embeddings = self.entity_embeddings(torch.LongTensor(head_ids))
            relation_embeddings = self.relation_embeddings(torch.LongTensor(relation_ids))
            scores = torch.norm(head_embeddings[:, None, :] + relation_embeddings[None, :, :] - self.entity_embeddings.weight.data, p=2, dim=-1)
            predicted_tails = torch.argmin(scores, dim=1).tolist()
            return predicted_tails
        elif head_ids is None and tail_ids is not None and relation_ids is not None:
            # 预测头实体
            tail_embeddings = self.entity_embeddings(torch.LongTensor(tail_ids))
            relation_embeddings = self.relation_embeddings(torch.LongTensor(relation_ids))
            scores = torch.norm(self.entity_embeddings.weight.data[None, :, :] + relation_embeddings[:, None, :] - tail_embeddings[:, None, :], p=2, dim=-1)
            predicted_heads = torch.argmin(scores, dim=1).tolist()
            return predicted_heads
        else:
            raise ValueError("Invalid input. Provide either head_ids and tail_ids to predict relation, head_ids and relation_ids to predict tail, or tail_ids and relation_ids to predict head.")


# 定义TransD模型
class TransD(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super(TransD, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # 初始化实体和关系的嵌入向量
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.normal_vectors = nn.Embedding(num_relations, embedding_dim)

        # 初始化嵌入向量
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        nn.init.xavier_uniform_(self.normal_vectors.weight.data)

    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        # 获取实体和关系的嵌入向量
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        
        normal_vector = self.normal_vectors(relations)
        
        head_proj = head_embeddings - torch.sum(head_embeddings * normal_vector, dim=-1, keepdim=True) * normal_vector
        tail_proj = tail_embeddings - torch.sum(tail_embeddings * normal_vector, dim=-1, keepdim=True) * normal_vector

        # 计算Trans模型中的损失函数
        scores = torch.norm(head_proj + relation_embeddings - tail_proj, p=2, dim=-1)
        return scores

    def predict(self, head_id: Optional[int] = None, tail_id: Optional[int] = None, relation_id: Optional[int] = None) -> Optional[int]:
        """
        预测函数，根据给定的头实体、尾实体或关系预测另一实体或关系

        Args:
        - head_id (Optional[int]): 头实体的索引
        - tail_id (Optional[int]): 尾实体的索引
        - relation_id (Optional[int]): 关系的索引

        Returns:
        - predicted_entity_or_relation (Optional[int]): 预测的实体或关系索引
        """
        if head_id is not None and tail_id is not None and relation_id is None:
            # 预测关系
            head_embedding = self.entity_embeddings(torch.tensor(head_id))
            tail_embedding = self.entity_embeddings(torch.tensor(tail_id))
            scores = torch.norm(head_embedding + self.relation_embeddings.weight.data - tail_embedding, p=2, dim=-1)
            predicted_relation = torch.argmin(scores).item()
            return predicted_relation
        elif head_id is not None and tail_id is None and relation_id is not None:
            # 预测尾实体
            head_embedding = self.entity_embeddings(torch.tensor(head_id))
            relation_embedding = self.relation_embeddings(torch.tensor(relation_id))
            scores = torch.norm(head_embedding + relation_embedding - self.entity_embeddings.weight.data, p=2, dim=-1)
            predicted_tail = torch.argmin(scores).item()
            return predicted_tail
        elif head_id is None and tail_id is not None and relation_id is not None:
            # 预测头实体
            tail_embedding = self.entity_embeddings(torch.tensor(tail_id))
            relation_embedding = self.relation_embeddings(torch.tensor(relation_id))
            scores = torch.norm(self.entity_embeddings.weight.data + relation_embedding - tail_embedding, p=2, dim=-1)
            predicted_head = torch.argmin(scores).item()
            return predicted_head
        else:
            raise ValueError("Invalid input. Provide either head_id and tail_id to predict relation, head_id and relation_id to predict tail, or tail_id and relation_id to predict head.")

    def batch_predict(self, head_ids: Optional[List[int]] = None, tail_ids: Optional[List[int]] = None, relation_ids: Optional[List[int]] = None) -> Optional[List[int]]:
        """
        批量预测函数，根据给定的头实体、尾实体或关系预测另一实体或关系

        Args:
        - head_ids (Optional[List[int]]): 头实体的索引列表
        - tail_ids (Optional[List[int]]): 尾实体的索引列表
        - relation_ids (Optional[List[int]]): 关系的索引列表

        Returns:
        - predicted_entities_or_relations (Optional[List[int]]): 预测的实体或关系索引列表
        """
        if head_ids is not None and tail_ids is not None and relation_ids is None:
            # 预测关系
            head_embeddings = self.entity_embeddings(torch.LongTensor(head_ids))
            tail_embeddings = self.entity_embeddings(torch.LongTensor(tail_ids))
            scores = torch.norm(head_embeddings[:, None, :] + self.relation_embeddings.weight.data - tail_embeddings[None, :, :], p=2, dim=-1)
            predicted_relations = torch.argmin(scores, dim=1).tolist()
            return predicted_relations
        elif head_ids is not None and tail_ids is None and relation_ids is not None:
            # 预测尾实体
            head_embeddings = self.entity_embeddings(torch.LongTensor(head_ids))
            relation_embeddings = self.relation_embeddings(torch.LongTensor(relation_ids))
            scores = torch.norm(head_embeddings[:, None, :] + relation_embeddings[None, :, :] - self.entity_embeddings.weight.data, p=2, dim=-1)
            predicted_tails = torch.argmin(scores, dim=1).tolist()
            return predicted_tails
        elif head_ids is None and tail_ids is not None and relation_ids is not None:
            # 预测头实体
            tail_embeddings = self.entity_embeddings(torch.LongTensor(tail_ids))
            relation_embeddings = self.relation_embeddings(torch.LongTensor(relation_ids))
            scores = torch.norm(self.entity_embeddings.weight.data[None, :, :] + relation_embeddings[:, None, :] - tail_embeddings[:, None, :], p=2, dim=-1)
            predicted_heads = torch.argmin(scores, dim=1).tolist()
            return predicted_heads
        else:
            raise ValueError("Invalid input. Provide either head_ids and tail_ids to predict relation, head_ids and relation_ids to predict tail, or tail_ids and relation_ids to predict head.")
        
    
# 定义TransR模型
class TransR(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super(TransR, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # 初始化实体和关系的嵌入向量
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.normal_vectors = nn.Embedding(num_relations, embedding_dim)

        # 初始化嵌入向量
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        nn.init.xavier_uniform_(self.normal_vectors.weight.data)

    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        # 获取实体和关系的嵌入向量
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        
        normal_vector = self.normal_vectors(relations)
        
        head_proj = head_embeddings - torch.sum(head_embeddings * normal_vector, dim=-1, keepdim=True) * normal_vector
        tail_proj = tail_embeddings - torch.sum(tail_embeddings * normal_vector, dim=-1, keepdim=True) * normal_vector

        # 计算Trans模型中的损失函数
        scores = torch.norm(head_proj + relation_embeddings - tail_proj, p=2, dim=-1)
        return scores

    def predict(self, head_id: Optional[int] = None, tail_id: Optional[int] = None, relation_id: Optional[int] = None) -> Optional[int]:
        """
        预测函数，根据给定的头实体、尾实体或关系预测另一实体或关系

        Args:
        - head_id (Optional[int]): 头实体的索引
        - tail_id (Optional[int]): 尾实体的索引
        - relation_id (Optional[int]): 关系的索引

        Returns:
        - predicted_entity_or_relation (Optional[int]): 预测的实体或关系索引
        """
        if head_id is not None and tail_id is not None and relation_id is None:
            # 预测关系
            head_embedding = self.entity_embeddings(torch.tensor(head_id))
            tail_embedding = self.entity_embeddings(torch.tensor(tail_id))
            scores = torch.norm(head_embedding + self.relation_embeddings.weight.data - tail_embedding, p=2, dim=-1)
            predicted_relation = torch.argmin(scores).item()
            return predicted_relation
        elif head_id is not None and tail_id is None and relation_id is not None:
            # 预测尾实体
            head_embedding = self.entity_embeddings(torch.tensor(head_id))
            relation_embedding = self.relation_embeddings(torch.tensor(relation_id))
            scores = torch.norm(head_embedding + relation_embedding - self.entity_embeddings.weight.data, p=2, dim=-1)
            predicted_tail = torch.argmin(scores).item()
            return predicted_tail
        elif head_id is None and tail_id is not None and relation_id is not None:
            # 预测头实体
            tail_embedding = self.entity_embeddings(torch.tensor(tail_id))
            relation_embedding = self.relation_embeddings(torch.tensor(relation_id))
            scores = torch.norm(self.entity_embeddings.weight.data + relation_embedding - tail_embedding, p=2, dim=-1)
            predicted_head = torch.argmin(scores).item()
            return predicted_head
        else:
            raise ValueError("Invalid input. Provide either head_id and tail_id to predict relation, head_id and relation_id to predict tail, or tail_id and relation_id to predict head.")

    def batch_predict(self, head_ids: Optional[List[int]] = None, tail_ids: Optional[List[int]] = None, relation_ids: Optional[List[int]] = None) -> Optional[List[int]]:
        """
        批量预测函数，根据给定的头实体、尾实体或关系预测另一实体或关系

        Args:
        - head_ids (Optional[List[int]]): 头实体的索引列表
        - tail_ids (Optional[List[int]]): 尾实体的索引列表
        - relation_ids (Optional[List[int]]): 关系的索引列表

        Returns:
        - predicted_entities_or_relations (Optional[List[int]]): 预测的实体或关系索引列表
        """
        if head_ids is not None and tail_ids is not None and relation_ids is None:
            # 预测关系
            head_embeddings = self.entity_embeddings(torch.LongTensor(head_ids))
            tail_embeddings = self.entity_embeddings(torch.LongTensor(tail_ids))
            scores = torch.norm(head_embeddings[:, None, :] + self.relation_embeddings.weight.data - tail_embeddings[None, :, :], p=2, dim=-1)
            predicted_relations = torch.argmin(scores, dim=1).tolist()
            return predicted_relations
        elif head_ids is not None and tail_ids is None and relation_ids is not None:
            # 预测尾实体
            head_embeddings = self.entity_embeddings(torch.LongTensor(head_ids))
            relation_embeddings = self.relation_embeddings(torch.LongTensor(relation_ids))
            scores = torch.norm(head_embeddings[:, None, :] + relation_embeddings[None, :, :] - self.entity_embeddings.weight.data, p=2, dim=-1)
            predicted_tails = torch.argmin(scores, dim=1).tolist()
            return predicted_tails
        elif head_ids is None and tail_ids is not None and relation_ids is not None:
            # 预测头实体
            tail_embeddings = self.entity_embeddings(torch.LongTensor(tail_ids))
            relation_embeddings = self.relation_embeddings(torch.LongTensor(relation_ids))
            scores = torch.norm(self.entity_embeddings.weight.data[None, :, :] + relation_embeddings[:, None, :] - tail_embeddings[:, None, :], p=2, dim=-1)
            predicted_heads = torch.argmin(scores, dim=1).tolist()
            return predicted_heads
        else:
            raise ValueError("Invalid input. Provide either head_ids and tail_ids to predict relation, head_ids and relation_ids to predict tail, or tail_ids and relation_ids to predict head.")