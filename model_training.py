import pandas as pd
from surprise import Reader, Dataset, KNNBasic, SVD
from surprise.model_selection import train_test_split
import pickle

# 加载数据
def load_data():
    user_ratings_df = pd.read_csv('data/user_ratings.csv')
    novels_df = pd.read_csv('data/novels.csv')
    return user_ratings_df, novels_df

# 准备Surprise数据
def prepare_surprise_data(user_ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(user_ratings_df[['user_id', 'novel_id', 'rating']], reader)
    return data

# 训练并保存模型
def train_and_save_models():
    user_ratings_df, _ = load_data()
    data = prepare_surprise_data(user_ratings_df)
    trainset = data.build_full_trainset()  # 使用完整训练集

    # 训练KNN模型
    sim_options = {'name': 'cosine', 'user_based': True}
    algo_knn = KNNBasic(sim_options=sim_options)
    algo_knn.fit(trainset)

    # 训练SVD模型
    algo_svd = SVD()
    algo_svd.fit(trainset)

    # 保存模型
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(algo_knn, f)
    with open('svd_model.pkl', 'wb') as f:
        pickle.dump(algo_svd, f)

if __name__ == '__main__':
    train_and_save_models()