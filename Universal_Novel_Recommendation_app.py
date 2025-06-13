import streamlit as st
import pandas as pd
from datetime import datetime
import pickle
import os

# 页面配置（宽屏 + 图标）
st.set_page_config(
    page_title="全平台小说个性化推荐系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义 CSS 注入（用 markdown 方式，避免文件依赖）
def inject_custom_css():
    custom_css = """
    <style>
        .main-header {
            font-size: 2.8rem !important; 
            color: #3498db;
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #e1e4e8;
        }
        .sub-header {
            font-size: 2rem !important;  
            color: #2c3e50;
            margin-top: 2.5rem !important;
        }
        .book-card {
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .book-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .book-title {
            font-size: 1.6rem !important;  
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
        }
        .book-meta {
            font-size: 1.1rem !important;  
            color: #6c757d !important;
            margin-bottom: 0.3rem !important;
        }
        .rating-stars {
            color: #f39c12;
            font-size: 1.2rem !important;
        }
        .feedback-text {
            font-size: 1.1rem !important;
            line-height: 1.6;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# 平台图标映射
PLATFORM_ICONS = {
    "微信读书": "logos/weixin_reading_logo.png",
    "QQ 阅读": "logos/qq_reading_logo.png",
    "Kindle 商店": "logos/kindle_store_logo.png",
    "番茄小说": "logos/tomato_novel_logo.png",
    "起点读书": "logos/qidian_reading_logo.png",
}

# 标签列表
ALL_TAGS = ["玄幻", "都市", "历史", "科幻", "悬疑", "武侠", "穿越", "奇幻", "冒险",
            "仙侠", "重生", "灵异", "战争", "言情", "搞笑", "无限流", "修真", "系统",
            "游戏", "娱乐", "异能", "权谋", "修仙", "校园", "江湖", "末世", "异世界"]

# 加载数据
def load_data():
    try:
        user_ratings_df = pd.read_csv('data/user_ratings.csv')
        novels_df = pd.read_csv('data/novels.csv')
        return user_ratings_df, novels_df
    except Exception as e:
        st.error(f"加载数据失败：{e}")
        return pd.DataFrame(), pd.DataFrame()

# 加载模型
def load_models():
    try:
        if not os.path.exists('svd_model.pkl'):
            st.error("模型文件 'svd_model.pkl' 不存在，请检查文件路径。")
            return None
        with open('svd_model.pkl', 'rb') as f:
            algo_svd = pickle.load(f)
        return algo_svd
    except Exception as e:
        st.error(f"加载模型失败：{e}，请检查模型文件是否损坏。")
        return None

# 模拟新用户数据输入
def get_new_user_data():
    st.subheader("基础信息", anchor=False)
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("性别", ("男", "女", "不想透露"), 
                             format_func=lambda x: "保密" if x == "不想透露" else x)
        birth_year = st.number_input("出生年份", min_value=1900, 
                                   max_value=datetime.now().year, value=2000,
                                   key="birth_year")
    with col2:
        occupation = st.selectbox("职业", ('学生', '上班族', '自由职业者', '退休', '不想透露'))
        reading_time = st.selectbox("每周阅读时长", (
            "几乎不阅读", "1-3小时", "4-6小时", "7-10小时", "10小时以上"
        ))
    
    st.subheader("阅读偏好", anchor=False)
    col1, col2 = st.columns(2)
    with col1:
        all_tags = ALL_TAGS + ["不想透露"]
        favorite_tags = st.multiselect("喜欢的标签", all_tags, 
                                     format_func=lambda x: "不透露" if x == "不想透露" else x)
    with col2:
        all_platforms = ["微信读书", "QQ 阅读", "Kindle 商店", "番茄小说", "起点读书", "不想透露"]
        preferred_platform = st.multiselect("常用平台", all_platforms,
                                          format_func=lambda x: "不透露" if x == "不想透露" else x)
    
    return gender, birth_year, occupation, reading_time, favorite_tags, preferred_platform

# 获取平台图标路径
def get_platform_icon(platform_name):
    for key, path in PLATFORM_ICONS.items():
        if key in platform_name or platform_name in key:
            if os.path.exists(path):
                return path
    return "logos/default_logo.png"  # 默认图标

# 基于用户画像生成偏好标签（内部逻辑，不展示）
def generate_preferred_tags(gender, birth_year, occupation, reading_time):
    preferred_tags = []
    # 性别偏好
    if gender == '男':
        preferred_tags.extend(['玄幻', '科幻', '武侠', '战争'])
    elif gender == '女':
        preferred_tags.extend(['言情', '校园', '都市', '重生'])
    # 年龄偏好
    age = datetime.now().year - birth_year
    if age < 20:  
        preferred_tags.extend(['校园', '修真', '异能', '搞笑'])
    elif age < 30:  
        preferred_tags.extend(['都市', '系统', '无限流', '职场'])
    elif age < 40:  
        preferred_tags.extend(['历史', '权谋', '战争', '悬疑'])
    else:  
        preferred_tags.extend(['历史', '现实', '职场', '文学'])
    # 职业偏好
    if occupation == '学生':
        preferred_tags.extend(['校园', '青春', '异能', '修真'])
    elif occupation == '上班族':
        preferred_tags.extend(['职场', '现实', '都市', '系统'])
    elif occupation == '自由职业者':
        preferred_tags.extend(['冒险', '奇幻', '武侠', '灵异'])
    elif occupation == '退休':
        preferred_tags.extend(['历史', '文学', '现实', '战争'])
    # 阅读时长偏好
    if reading_time == "几乎不阅读":  
        preferred_tags.extend(['短篇', '言情', '搞笑', '校园'])
    elif reading_time == "1-3小时":  
        preferred_tags.extend(['都市', '修真', '异能', '悬疑'])
    elif reading_time == "4-6小时":  
        preferred_tags.extend(['玄幻', '武侠', '无限流', '历史'])
    elif reading_time == "7-10小时" or reading_time == "10小时以上":  
        preferred_tags.extend(['长篇', '史诗', '系统', '修仙'])
    
    # 统计高频标签（取前4个）
    tag_counts = {}
    for tag in preferred_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    return [tag for tag, _ in sorted_tags[:4]]

# SVD 协同过滤推荐
def svd_recommendations(algo_svd, novels_df, user_ratings_df, n=100):
    if algo_svd is None or novels_df.empty:
        return []
    predictions = []
    for _, novel in novels_df.iterrows():
        novel_id = novel['id']
        pred = algo_svd.predict(-1, novel_id)
        platform_rating = novel['rating'] if 'rating' in novel else 0
        predictions.append({
            'id': novel_id,
            'title': novel['title'],
            'author': novel['author'],
            'tags': novel['tags'],
            'platform': novel['platform'],
            'platform_icon': get_platform_icon(novel['platform']),
            'predicted_rating': round(pred.est, 2),
            'platform_rating': round(platform_rating, 2) if platform_rating > 0 else "暂无评分"
        })
    # 按预测评分排序
    predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
    return predictions[:n]

# 内容推荐（标签匹配）
def content_based_recommendations(novels_df, preferred_tags, n=50):
    if novels_df.empty:
        return []
    matching_novels = []
    for _, novel in novels_df.iterrows():
        match_count = sum(1 for tag in preferred_tags if tag in str(novel['tags']))
        if match_count > 0:
            platform_rating = novel['rating'] if 'rating' in novel else 0
            matching_novels.append({
                'id': novel['id'],
                'title': novel['title'],
                'author': novel['author'],
                'tags': novel['tags'],
                'platform': novel['platform'],
                'platform_icon': get_platform_icon(novel['platform']),
                'match_score': match_count,
                'platform_rating': platform_rating
            })
    # 按匹配度排序
    matching_novels.sort(key=lambda x: x['match_score'], reverse=True)
    return matching_novels[:n]

# 混合推荐（协同过滤 + 内容推荐）
def hybrid_recommendations(algo_svd, novels_df, user_ratings_df, preferred_tags, n=100):
    if novels_df.empty:
        return []
    # 1. 协同过滤结果
    cf_recs = svd_recommendations(algo_svd, novels_df, user_ratings_df, n)
    # 2. 内容推荐结果
    content_recs = content_based_recommendations(novels_df, preferred_tags, n)
    
    # 3. 融合逻辑（内容推荐优先，去重后合并）
    hybrid_result = []
    content_ids = {rec['id'] for rec in content_recs}
    # 先加内容推荐（取前50）
    for rec in content_recs[:50]:
        hybrid_result.append({
            'id': rec['id'],
            'title': rec['title'],
            'author': rec['author'],
            'tags': rec['tags'],
            'platform': rec['platform'],
            'platform_icon': rec['platform_icon'],
            'predicted_rating': 0,  
            'platform_rating': round(rec['platform_rating'], 2) if rec['platform_rating'] > 0 else "暂无评分",
            'match_score': rec['match_score']
        })
    # 再加协同过滤（补到100，跳过已存在的）
    cf_count = 0
    for rec in cf_recs:
        if rec['id'] not in content_ids and cf_count < 50:
            hybrid_result.append({
                'id': rec['id'],
                'title': rec['title'],
                'author': rec['author'],
                'tags': rec['tags'],
                'platform': rec['platform'],
                'platform_icon': rec['platform_icon'],
                'predicted_rating': rec['predicted_rating'],
                'platform_rating': rec['platform_rating'],
                'match_score': 0
            })
            cf_count += 1
    
    # 4. 计算最终评分（内容匹配度70% + 协同过滤30% + 平台评分校准）
    if hybrid_result:
        max_match = max(rec['match_score'] for rec in hybrid_result) or 1
        # 检查是否有 predicted_rating 大于 0 的元素
        cf_ratings = [rec['predicted_rating'] for rec in hybrid_result if rec['predicted_rating'] > 0]
        max_cf = max(cf_ratings) if cf_ratings else 5
        for rec in hybrid_result:
            # 基础分计算
            if rec['match_score'] > 0:
                content_score = (rec['match_score'] / max_match) * 5
                base = content_score * 0.7 + 3 * 0.3
            else:
                base = rec['predicted_rating']
            # 平台评分校准
            if isinstance(rec['platform_rating'], (int, float)) and rec['platform_rating'] > 0:
                if rec['platform_rating'] > 3:
                    adj = ((rec['platform_rating'] - 3) / 0.2) * 0.03
                else:
                    adj = ((3 - rec['platform_rating']) / 0.2) * (-0.01)
                final = min(5, max(0, base + adj))
            else:
                final = base
            rec['final_score'] = final
        # 按最终分排序
        hybrid_result.sort(key=lambda x: x['final_score'], reverse=True)
    
    # 格式化输出
    return [
        {
            'id': rec['id'],
            'title': rec['title'],
            'author': rec['author'],
            'tags': rec['tags'],
            'platform': rec['platform'],
            'platform_icon': rec['platform_icon'],
            'predicted_rating': round(rec['final_score'], 2),
            'platform_rating': rec['platform_rating']
        } 
        for rec in hybrid_result[:n]
    ]

# 导航步骤显示
def show_step_nav(current_step):
    steps = ["填写信息", "查看推荐", "反馈评价"]
    cols = st.columns(3)
    for i, step in enumerate(steps):
        with cols[i]:
            if i + 1 == current_step:
                st.markdown(
                    f"<h3 style='color: #3498db; text-align: center; font-weight: 700;'>{i+1}. {step}</h3>"
                    f"<div style='height: 3px; width: 60%; background-color: #3498db; margin: 0.5rem auto;'></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<h3 style='color: #95a5a6; text-align: center; font-weight: 500;'>{i+1}. {step}</h3>"
                    f"<div style='height: 3px; width: 60%; background-color: #e1e4e8; margin: 0.5rem auto;'></div>",
                    unsafe_allow_html=True
                )

# 主流程
def main():
    inject_custom_css()  # 注入自定义样式
    st.markdown("<h1 class='main-header'>小说推荐系统</h1>", unsafe_allow_html=True)
    
    # 初始化会话状态
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'satisfaction' not in st.session_state:
        st.session_state.satisfaction = 5

    # 显示步骤导航
    show_step_nav(st.session_state.current_step)

    # 步骤1：填写信息
    if st.session_state.current_step == 1:
        with st.container():
            st.markdown("<h2 class='sub-header'>1. 完善你的阅读画像</h2>", unsafe_allow_html=True)
            st.info("填写信息后，系统会生成专属你的小说推荐单", icon="✨")
            
            # 获取用户输入
            gender, birth_year, occupation, reading_time, favorite_tags, preferred_platform = get_new_user_data()
            
            # 保存用户数据
            st.session_state.user_data = {
                'gender': gender,
                'birth_year': birth_year,
                'occupation': occupation,
                'reading_time': reading_time,
                'favorite_tags': favorite_tags,
                'preferred_platform': preferred_platform
            }
            
            # 生成偏好标签（内部逻辑，不展示）
            preferred_tags = generate_preferred_tags(
                gender, birth_year, occupation, reading_time
            )
            st.session_state.preferred_tags = preferred_tags  # 暂存
            
            if st.button("生成专属推荐", type="primary", help="点击后系统将基于您的偏好生成个性化推荐"):
                # 加载数据和模型
                user_ratings_df, novels_df = load_data()
                algo_svd = load_models()
                
                # 生成混合推荐
                st.session_state.recommendations = hybrid_recommendations(
                    algo_svd, novels_df, user_ratings_df, 
                    preferred_tags, n=100
                )
                
                # 进入下一步
                st.session_state.current_step = 2
                st.rerun()
    
    # 步骤2：查看推荐
    elif st.session_state.current_step == 2:
        if not st.session_state.recommendations:
            st.warning("请先填写信息并生成推荐")
            if st.button("返回填写信息", type="secondary"):
                st.session_state.current_step = 1
                st.rerun()
            return
        
        with st.container():
            st.markdown("<h2 class='sub-header'>2. 探索推荐小说</h2>", unsafe_allow_html=True)
            st.success("根据您的阅读偏好，为您精心推荐以下小说", icon="📚")
            
            # 初始化分页状态
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1
            
            # 计算当前页显示的书籍
            recommendations = st.session_state.recommendations
            books_per_page = 8
            start_idx = (st.session_state.current_page - 1) * books_per_page
            end_idx = min(start_idx + books_per_page, len(recommendations))
            current_books = recommendations[start_idx:end_idx]
            
            st.write(f"为您推荐的小说（共 {len(recommendations)} 本）：")
            
            # 显示书籍卡片（增加文字大小）
            for book in current_books:
                with st.container():
                    col1, col2 = st.columns([1, 4], gap="medium")
                    with col1:
                        # 替换 use_column_width 为 use_container_width
                        if os.path.exists(book['platform_icon']):
                            st.image(book['platform_icon'], width=80, 
                                   caption=book['platform'], use_container_width=False)
                        else:
                            st.image("logos/default_logo.png", width=80, 
                                   caption="未知平台", use_container_width=False)
                    
                    with col2:
                        st.markdown(f"<h3 class='book-title'>《{book['title']}》</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p class='book-meta'><strong>作者:</strong> {book['author']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='book-meta'><strong>类型:</strong> {book['tags']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='book-meta'><strong>平台评分:</strong> ⭐️ {book['platform_rating']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class=''><strong>推荐评分:</strong> <span class='rating-stars'>{book['predicted_rating']}/5.0</span></p>", unsafe_allow_html=True)
                    
                    st.markdown("---")
            
            # 分页控制
            total_pages = (len(recommendations) + books_per_page - 1) // books_per_page
            if total_pages > 1:
                st.markdown("<div class='page-nav'>", unsafe_allow_html=True)
                
                if st.session_state.current_page > 1:
                    if st.button("上一页", key="prev_page", 
                              help="查看前一页推荐的小说", 
                              disabled=st.session_state.current_page <= 1):
                        st.session_state.current_page -= 1
                        st.rerun()
                
                st.markdown(f"<span style='margin: 0 1rem;'>第 {st.session_state.current_page} 页 / 共 {total_pages} 页</span>", unsafe_allow_html=True)
                
                if st.session_state.current_page < total_pages:
                    if st.button("下一页", key="next_page", 
                              help="查看下一页推荐的小说",
                              disabled=st.session_state.current_page >= total_pages):
                        st.session_state.current_page += 1
                        st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # 进入下一步
            if st.button("前往满意度评价", type="primary",
                       help="对推荐结果进行评价，帮助我们优化系统"):
                st.session_state.current_step = 3
                st.rerun()
    
    # 步骤3：满意度反馈
    elif st.session_state.current_step == 3:
        if not st.session_state.recommendations:
            st.warning("请先填写信息并生成推荐")
            if st.button("返回填写信息", type="secondary"):
                st.session_state.current_step = 1
                st.rerun()
            return
        
        with st.container():
            st.markdown("<h2 class='sub-header'>3. 您的反馈对我们很重要</h2>", unsafe_allow_html=True)
            st.info("您的每一条评价都将帮助我们优化推荐算法，提升阅读体验", icon="💬")
            
            st.write("感谢您使用我们的推荐系统！请对本次推荐结果进行评分：")
            
            # 满意度评分
            st.session_state.satisfaction = st.slider(
                "请对推荐结果进行评分 (1-10)", 
                1, 10, 
                st.session_state.satisfaction,
                format="%d 分"
            )
            
            # 额外反馈
            feedback = st.text_area("您的其他建议或反馈（可选）", 
                                   placeholder="例如：希望增加更多科幻类小说推荐...",
                                   height=100)
            
            if st.button("提交反馈", type="primary",
                       help="提交您的反馈意见，帮助我们持续改进"):
                # 根据评分显示不同的反馈信息
                if st.session_state.satisfaction >= 8:
                    st.success("🌟 感谢您的认可！我们会持续优化推荐算法，为您发现更多精彩好书～")
                    st.markdown("<p class='feedback-text'>您的支持是我们进步的动力！后续会基于您的偏好，推送更多符合您口味的小说。</p>", unsafe_allow_html=True)
                elif st.session_state.satisfaction >= 5:
                    st.info("👍 您的反馈已收到，我们会根据您的意见继续改进！")
                    st.markdown("<p class='feedback-text'>我们注意到推荐结果还有提升空间，会分析您的偏好数据，优化算法模型。</p>", unsafe_allow_html=True)
                else:
                    st.warning("😔 抱歉没能满足您的期望，我们会仔细分析原因并优化推荐算法～")
                    st.markdown("<p class='feedback-text'>您的详细反馈对我们非常重要，请放心，我们会持续优化模型，下次为您提供更精准的推荐。</p>", unsafe_allow_html=True)
                
                # 保存反馈数据（实际应用中可保存到数据库）
                feedback_data = {
                    'satisfaction': st.session_state.satisfaction,
                    'feedback': feedback,
                    'user_data': st.session_state.user_data,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # 这里可以添加保存反馈数据的代码
                # 例如：save_feedback(feedback_data)
                
                st.write("您的反馈已提交，感谢您的支持！")
                
    
    # 页脚
    st.markdown("<div class='footer'>© 全平台小说推荐系统 | 为您发现更多好书</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()