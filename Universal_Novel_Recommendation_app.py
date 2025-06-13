import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from fastai.collab import CollabDataLoaders, collab_learner, load_learner
from fastai.data.transforms import ColReader
import base64
import io

# 设置页面配置
st.set_page_config(
    page_title="小说推荐系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        color: #1a365d;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #2d3748;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f7fafc;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .book-cover {
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
        width: 100%;
        height: auto;
    }
    .book-title {
        font-weight: bold;
        margin-bottom: 0.25rem;
    }
    .book-author {
        color: #718096;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }
    .rating-stars {
        color: #ecc94b;
        margin-bottom: 0.5rem;
    }
    .tag {
        display: inline-block;
        background-color: #edf2f7;
        color: #4a5568;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        margin-right: 0.25rem;
        margin-bottom: 0.25rem;
        font-size: 0.75rem;
    }
    .recommendation-header {
        color: #2a4365;
        font-weight: bold;
        margin-top: 1rem;
    }
    .no-results {
        color: #718096;
        text-align: center;
        padding: 2rem;
    }
    .platform-icon {
        vertical-align: middle;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 加载数据函数
@st.cache_data
def load_data():
    try:
        # 尝试加载小说数据
        if os.path.exists('novels_data.csv'):
            novels_df = pd.read_csv('novels_data.csv')
        else:
            # 创建示例数据
            novels_df = pd.DataFrame({
                'id': range(1, 101),
                'title': [f'小说{i}' for i in range(1, 101)],
                'author': [f'作者{i%10+1}' for i in range(1, 101)],
                'tags': ['科幻,冒险', '言情,校园', '武侠,玄幻', '悬疑,推理', '历史,穿越'] * 20,
                'platform': ['起点中文网', '晋江文学城', '番茄小说', 'QQ阅读', '微信读书'] * 20,
                'rating': np.random.uniform(3.5, 5.0, 100).round(1),
                'cover_url': [f'https://picsum.photos/seed/book{i}/200/300' for i in range(1, 101)]
            })
            
            # 为平台添加图标URL
            platform_icons = {
                '起点中文网': 'https://picsum.photos/seed/qidian/80/80',
                '晋江文学城': 'https://picsum.photos/seed/jj/80/80',
                '番茄小说': 'https://picsum.photos/seed/fanqie/80/80',
                'QQ阅读': 'https://picsum.photos/seed/qq/80/80',
                '微信读书': 'https://picsum.photos/seed/wx/80/80'
            }
            novels_df['platform_icon'] = novels_df['platform'].map(platform_icons)
            
            # 保存示例数据
            novels_df.to_csv('novels_data.csv', index=False)
        
        # 尝试加载用户评分数据
        if os.path.exists('user_ratings.csv'):
            user_ratings_df = pd.read_csv('user_ratings.csv')
        else:
            # 创建示例评分数据
            user_ids = list(range(-10, 0))  # 假设有10个用户
            novel_ids = novels_df['id'].tolist()
            data = []
            
            for user_id in user_ids:
                # 每个用户随机评价5-10本小说
                num_ratings = np.random.randint(5, 11)
                rated_novels = np.random.choice(novel_ids, num_ratings, replace=False)
                
                for novel_id in rated_novels:
                    rating = np.random.uniform(3.0, 5.0).round(1)
                    data.append({
                        'user_id': user_id,
                        'novel_id': novel_id,
                        'rating': rating
                    })
            
            user_ratings_df = pd.DataFrame(data)
            user_ratings_df.to_csv('user_ratings.csv', index=False)
        
        return novels_df, user_ratings_df
    except Exception as e:
        st.error(f"加载数据失败: {e}")
        # 创建空数据框
        return pd.DataFrame(), pd.DataFrame()

# 加载模型函数
@st.cache_resource
def load_model():
    try:
        if os.path.exists('novel_recommendation_model.pkl'):
            # 加载保存的模型
            with open('novel_recommendation_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            # 创建示例模型
            st.info("未找到模型文件，正在创建示例模型...")
            # 这里应该是实际的模型训练代码
            # 为了演示，返回一个简单的函数
            def dummy_model(user_id, novel_id):
                # 返回随机评分作为示例
                return np.random.uniform(3.0, 5.0)
            return dummy_model
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None

# 处理平台图标，增加错误处理
def get_platform_icon(platform):
    platform_icons = {
        '起点中文网': 'https://picsum.photos/seed/qidian/80/80',
        '晋江文学城': 'https://picsum.photos/seed/jj/80/80',
        '番茄小说': 'https://picsum.photos/seed/fanqie/80/80',
        'QQ阅读': 'https://picsum.photos/seed/qq/80/80',
        '微信读书': 'https://picsum.photos/seed/wx/80/80'
    }
    
    # 获取默认图标（当平台不在字典中时使用）
    default_icon = 'https://picsum.photos/seed/default/80/80'
    
    return platform_icons.get(platform, default_icon)

# 显示图标，增加错误处理
def display_icon(icon_url, width=80):
    try:
        # 尝试直接显示图标
        st.image(icon_url, width=width)
    except Exception as e:
        # 显示错误信息和默认图标
        st.write(f"无法加载图标: {e}")
        st.image('https://picsum.photos/seed/error/80/80', width=width)

# 推荐函数
def get_recommendations(model, novels_df, user_ratings_df, user_id, n=10):
    if model is None:
        st.warning("模型未加载，无法生成推荐")
        return []
    
    # 获取用户已评分的小说ID
    if user_id in user_ratings_df['user_id'].values:
        user_rated_novels = set(user_ratings_df[user_ratings_df['user_id'] == user_id]['novel_id'].values)
    else:
        user_rated_novels = set()
    
    # 为未评分的小说生成预测评分
    predictions = []
    for _, novel in novels_df.iterrows():
        novel_id = novel['id']
        if novel_id not in user_rated_novels:
            try:
                # 尝试使用模型预测评分
                if callable(model):
                    # 如果模型是一个函数（如示例模型）
                    predicted_rating = model(user_id, novel_id)
                else:
                    # 如果模型是一个fastai模型
                    predicted_rating = model.predict((user_id, novel_id))[1].item()
                
                platform_rating = novel['rating'] if 'rating' in novel else 0
                
                predictions.append({
                    'id': novel_id,
                    'title': novel['title'],
                    'author': novel['author'],
                    'tags': novel['tags'],
                    'platform': novel['platform'],
                    'platform_icon': novel.get('platform_icon', get_platform_icon(novel['platform'])),
                    'predicted_rating': round(predicted_rating, 2),
                    'platform_rating': round(platform_rating, 2) if platform_rating > 0 else "暂无评分",
                    'cover_url': novel['cover_url']
                })
            except Exception as e:
                st.write(f"预测小说 {novel['title']} 评分时出错: {e}")
    
    # 按预测评分排序
    predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
    
    return predictions[:n]

# 主函数
def main():
    # 页面标题
    st.markdown("<h1 class='main-header'>小说推荐系统</h1>", unsafe_allow_html=True)
    
    # 加载数据和模型
    novels_df, user_ratings_df = load_data()
    model = load_model()
    
    # 侧边栏
    st.sidebar.header("用户设置")
    
    # 用户ID选择（为了演示，只有-1到-10的用户ID）
    user_id = st.sidebar.selectbox(
        "选择用户",
        options=list(range(-1, -11, -1)),
        index=0,
        format_func=lambda x: f"用户 {-x}"
    )
    
    # 筛选选项
    st.sidebar.header("推荐筛选")
    
    # 平台筛选
    all_platforms = sorted(novels_df['platform'].unique().tolist()) if not novels_df.empty else []
    selected_platforms = st.sidebar.multiselect(
        "平台",
        options=all_platforms,
        default=all_platforms
    )
    
    # 标签筛选
    all_tags = set()
    if not novels_df.empty:
        for tags in novels_df['tags'].str.split(',').dropna():
            all_tags.update([tag.strip() for tag in tags])
    all_tags = sorted(list(all_tags))
    
    selected_tags = st.sidebar.multiselect(
        "标签",
        options=all_tags,
        default=[]
    )
    
    # 评分筛选
    rating_min, rating_max = st.sidebar.slider(
        "最低平台评分",
        min_value=0.0,
        max_value=5.0,
        value=(3.0, 5.0),
        step=0.1
    )
    
    # 推荐数量
    num_recommendations = st.sidebar.slider(
        "推荐数量",
        min_value=5,
        max_value=50,
        value=15,
        step=5
    )
    
    # 筛选数据
    filtered_novels = novels_df.copy()
    
    if selected_platforms:
        filtered_novels = filtered_novels[filtered_novels['platform'].isin(selected_platforms)]
    
    if selected_tags:
        # 筛选包含至少一个所选标签的小说
        filtered_novels = filtered_novels[
            filtered_novels['tags'].apply(
                lambda x: any(tag in x for tag in selected_tags) if isinstance(x, str) else False
            )
        ]
    
    if 'rating' in filtered_novels.columns:
        filtered_novels = filtered_novels[
            (filtered_novels['rating'] >= rating_min) & 
            (filtered_novels['rating'] <= rating_max)
        ]
    
    # 生成推荐
    recommendations = get_recommendations(model, filtered_novels, user_ratings_df, user_id, num_recommendations)
    
    # 显示推荐结果
    st.markdown(f"<h2 class='sub-header'>为您推荐的小说</h2>", unsafe_allow_html=True)
    
    if not recommendations:
        st.markdown("<div class='no-results'>没有找到符合条件的推荐。请尝试调整筛选条件。</div>", unsafe_allow_html=True)
    else:
        # 创建网格布局
        cols = st.columns(3)
        
        for i, book in enumerate(recommendations):
            col = cols[i % 3]
            with col:
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    
                    # 显示封面
                    st.image(book['cover_url'], caption=book['title'], width=200, use_column_width=True)
                    
                    # 显示平台图标
                    st.markdown(f"<div class='platform-icon'>", unsafe_allow_html=True)
                    display_icon(book['platform_icon'], width=40)
                    st.markdown(f"{book['platform']}</div>", unsafe_allow_html=True)
                    
                    # 显示标题和作者
                    st.markdown(f"<div class='book-title'>{book['title']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='book-author'>作者: {book['author']}</div>", unsafe_allow_html=True)
                    
                    # 显示评分
                    predicted_stars = "★" * int(book['predicted_rating']) + "☆" * (5 - int(book['predicted_rating']))
                    st.markdown(f"<div class='rating-stars'>预测评分: {book['predicted_rating']} {predicted_stars}</div>", unsafe_allow_html=True)
                    
                    if book['platform_rating'] != "暂无评分":
                        platform_stars = "★" * int(book['platform_rating']) + "☆" * (5 - int(book['platform_rating']))
                        st.markdown(f"<div class='rating-stars'>平台评分: {book['platform_rating']} {platform_stars}</div>", unsafe_allow_html=True)
                    
                    # 显示标签
                    if isinstance(book['tags'], str):
                        st.markdown("<div>标签: </div>", unsafe_allow_html=True)
                        for tag in book['tags'].split(','):
                            st.markdown(f"<span class='tag'>{tag.strip()}</span>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    
    # 数据统计和可视化
    if not st.sidebar.checkbox("隐藏数据统计", False):
        st.markdown(f"<h2 class='sub-header'>数据统计</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>小说平台分布</h3>", unsafe_allow_html=True)
            if not novels_df.empty and 'platform' in novels_df.columns:
                platform_counts = novels_df['platform'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=platform_counts.index, y=platform_counts.values, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig)
            else:
                st.write("暂无平台数据")
        
        with col2:
            st.markdown("<h3>评分分布</h3>", unsafe_allow_html=True)
            if not novels_df.empty and 'rating' in novels_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(novels_df['rating'], bins=10, kde=True, ax=ax)
                st.pyplot(fig)
            else:
                st.write("暂无评分数据")

if __name__ == "__main__":
    main()
