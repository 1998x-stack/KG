import json
import pandas as pd

class DataProcessor:
    def __init__(self, train_file, valid_file, test_file, save_dir, dict_dir):
        """
        数据处理类，用于处理数据集

        Args:
            train_file (str): 训练集文件路径
            valid_file (str): 验证集文件路径
            test_file (str): 测试集文件路径
            save_dir (str): 保存文件的目录路径
            dict_dir (str): 字典文件的目录路径
        """
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.save_dir = save_dir
        self.dict_dir = dict_dir

    def transform_to_standard(self, raw_path, save_path):
        """
        将原始数据转换为标准格式，并保存到指定路径

        Args:
            raw_path (str): 原始数据文件路径
            save_path (str): 转换后的数据保存路径
        """
        df = pd.read_csv(raw_path, header=None, names=["head", "relation", "tail"], sep="\t", encoding="utf-8")
        df.to_csv(save_path, header=None, index=None, sep="\t", encoding="utf-8")

    def generate_dict(self, data_paths):
        """
        生成实体和关系的字典，并保存到指定路径

        Args:
            data_paths (list): 数据文件路径列表
        """
        df = pd.concat([pd.read_csv(path, header=None, names=["head", "relation", "tail"], sep="\t", encoding="utf-8") for path in data_paths])
        entity_dict = {entity: idx for idx, entity in enumerate(list(set(df["head"].unique().tolist() + df["tail"].unique().tolist())))}
        relation_dict = {relation: idx for idx, relation in enumerate(df["relation"].unique().tolist())}
        with open(self.dict_dir + "entity_dict.json", "w", encoding="utf-8") as f:
            json.dump(entity_dict, f, ensure_ascii=False, indent=4)
        with open(self.dict_dir + "relation_dict.json", "w", encoding="utf-8") as f:
            json.dump(relation_dict, f, ensure_ascii=False, indent=4)

    def split_data(self):
        """
        将数据集拆分为训练集、验证集和测试集，并保存到指定路径
        """
        train_df = pd.read_csv(self.train_file, header=None, names=["head", "relation", "tail"], sep="\t", encoding="utf-8")
        valid_df = pd.read_csv(self.valid_file, header=None, names=["head", "relation", "tail"], sep="\t", encoding="utf-8")
        test_df = pd.read_csv(self.test_file, header=None, names=["head", "relation", "tail"], sep="\t", encoding="utf-8")
        train_df.to_csv(self.save_dir + "train.txt", header=None, index=None, sep="\t", encoding="utf-8")
        valid_df.to_csv(self.save_dir + "valid.txt", header=None, index=None, sep="\t", encoding="utf-8")
        test_df.to_csv(self.save_dir + "test.txt", header=None, index=None, sep="\t", encoding="utf-8")

    def process_data(self):
        """
        数据处理的主流程，包括转换数据格式、生成字典和拆分数据集
        """
        self.transform_to_standard(raw_path=self.train_file, save_path=self.save_dir + "train.txt")
        self.transform_to_standard(raw_path=self.valid_file, save_path=self.save_dir + "valid.txt")
        self.transform_to_standard(raw_path=self.test_file, save_path=self.save_dir + "test.txt")
        self.generate_dict(data_paths=[self.train_file, self.valid_file, self.test_file])
        self.split_data()

if __name__ == "__main__":
    trainFile = "./FB15k/freebase_mtr100_mte100-train.txt"
    validFile = "./FB15k/freebase_mtr100_mte100-valid.txt"
    testFile = "./FB15k/freebase_mtr100_mte100-test.txt"
    saveDir = "./FB15k/"
    dictDir = "./FB15k/"

    data_processor = DataProcessor(train_file=trainFile, valid_file=validFile, test_file=testFile, save_dir=saveDir, dict_dir=dictDir)
    data_processor.process_data()