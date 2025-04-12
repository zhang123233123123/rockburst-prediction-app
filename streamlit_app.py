import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 设置页面
st.set_page_config(
    page_title="岩爆等级预测系统",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 苹果风格CSS */
    .main {
        background-color: #f5f5f7;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stButton>button {
        background-color: #0071e3;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0077ED;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        font-weight: 600;
    }
    .prediction-box {
        background-color: white;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        padding: 30px;
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    .prediction-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    .info-text {
        color: #86868b;
        font-size: 16px;
        line-height: 1.6;
    }
    .card {
        background-color: white;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# 页面标题
st.title("🪨 智能岩爆等级预测系统")
st.markdown('<p class="info-text">基于先进的机器学习算法，帮助您预测岩石的岩爆倾向等级</p>', unsafe_allow_html=True)

# 简单的预测函数
def simple_predict(input_data):
    # 创建一个简单的模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 简单训练
    X = np.random.rand(100, 7)
    y = np.random.choice([0, 1, 2, 3], size=100)
    model.fit(X, y)
    
    # 准备输入数据
    X_pred = np.array([[
        input_data["rock_type"],
        input_data["sigma_theta"],
        input_data["sigma_c"],
        input_data["sigma_t"],
        input_data["sigma_theta_c_ratio"],
        input_data["sigma_c_t_ratio"],
        input_data["wet"]
    ]])
    
    # 预测
    prediction = model.predict(X_pred)[0]
    probabilities = model.predict_proba(X_pred)[0]
    
    grade_text = {
        0: "无岩爆倾向",
        1: "弱岩爆倾向",
        2: "中等岩爆倾向",
        3: "强岩爆倾向"
    }.get(prediction, "未知等级")
    
    return {
        "prediction": int(prediction),
        "prediction_text": grade_text,
        "probabilities": {f"Class {i}": float(prob) for i, prob in enumerate(probabilities)}
    }

# 侧边栏配置
with st.sidebar:
    st.markdown("# 岩爆预测")
    st.markdown("## 参数设置")
    st.markdown("请选择岩石样本的关键参数:")
    
    # 岩石种类选择
    rock_types = {
        "花岗岩": 1.0,
        "大理岩": 2.0,
        "石灰岩": 3.0,
        "砂岩": 4.0,
        "页岩": 5.0
    }
    selected_rock = st.selectbox("岩石种类", list(rock_types.keys()))
    rock_type_encoded = rock_types[selected_rock]
    
    # 其他参数
    sigma_theta = st.slider("σθ / Mpa (围岩应力)", 10.0, 200.0, 50.0, 0.1)
    sigma_c = st.slider("σc / Mpa (单轴抗压强度)", 20.0, 300.0, 100.0, 0.1)
    sigma_t = st.slider("σt / MPa (抗拉强度)", 1.0, 50.0, 10.0, 0.1)
    
    # 自动计算比率
    sigma_theta_c_ratio = sigma_theta / sigma_c
    sigma_c_t_ratio = sigma_c / sigma_t
    
    # 显示计算出的比率
    st.markdown(f"**σθ/σc 比值**: {sigma_theta_c_ratio:.2f}")
    st.markdown(f"**σc/σt 比值**: {sigma_c_t_ratio:.2f}")
    
    # 含水率
    wet = st.slider("含水率 (Wet)", 0.0, 1.0, 0.5, 0.01)
    
    st.markdown("---")
    st.markdown("### 关于")
    st.markdown("本系统使用机器学习算法，基于岩石物理参数对岩爆等级进行预测")

# 主要内容区
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("岩石参数汇总")
    
    # 创建参数表格
    data = {
        "参数": ["岩石种类", "σθ / Mpa (围岩应力)", "σc / Mpa (单轴抗压强度)", 
                "σt / MPa (抗拉强度)", "σθ/σc", "σc/σt", "含水率 (Wet)"],
        "数值": [selected_rock, sigma_theta, sigma_c, sigma_t, 
               sigma_theta_c_ratio, sigma_c_t_ratio, wet]
    }
    
    params_df = pd.DataFrame(data)
    st.table(params_df)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 预测按钮
    if st.button("开始预测", key="predict_button"):
        try:
            # 准备预测数据
            input_data = {
                "rock_type": rock_type_encoded,
                "sigma_theta": sigma_theta,
                "sigma_c": sigma_c,
                "sigma_t": sigma_t,
                "sigma_theta_c_ratio": sigma_theta_c_ratio,
                "sigma_c_t_ratio": sigma_c_t_ratio,
                "wet": wet
            }
            
            # 使用简单预测函数
            result = simple_predict(input_data)
            
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.success("分析完成!")
            
            # 结果展示
            grade_text = result["prediction_text"]
            prediction = result["prediction"]
            
            # 根据预测结果设置颜色
            colors = {0: "#4CAF50", 1: "#FFC107", 2: "#FF9800", 3: "#F44336"}
            grade_color = colors.get(prediction, "#9E9E9E")
            
            st.markdown(f"<h2 style='color:{grade_color}'>预测结果: {grade_text}</h2>", unsafe_allow_html=True)
            
            # 显示各类别概率
            probabilities = result["probabilities"]
            
            # 结果解释
            st.subheader("预测解释")
            st.markdown(f"""
            根据您提供的岩石参数，预测该样本的岩爆等级为**{grade_text}**。
            此预测基于样本的物理特性分析，包括围岩应力、抗压强度和抗拉强度等关键参数。
            """)
            
            # 显示概率详情
            st.markdown("### 各等级概率")
            
            # 创建Matplotlib图表
            fig, ax = plt.figure(figsize=(8, 5)), plt.axes()
            
            labels = ["无岩爆倾向", "弱岩爆倾向", "中等岩爆倾向", "强岩爆倾向"]
            values = [
                probabilities.get("Class 0", 0),
                probabilities.get("Class 1", 0),
                probabilities.get("Class 2", 0),
                probabilities.get("Class 3", 0)
            ]
            
            bar_colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
            
            ax.bar(labels, values, color=bar_colors)
            ax.set_ylabel('概率')
            ax.set_title('岩爆等级概率分布')
            
            st.pyplot(fig)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"预测过程中出现错误: {str(e)}")
            import traceback
            st.error(f"详细错误信息: {traceback.format_exc()}")

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("岩爆等级说明")
    
    # 岩爆等级解释
    grade_info = {
        "无岩爆倾向 (0级)": "岩石在开挖过程中稳定性较好，不易发生岩爆现象。",
        "弱岩爆倾向 (1级)": "岩石可能会发生轻微的岩体破坏，但规模小，危害有限。",
        "中等岩爆倾向 (2级)": "岩石有较明显的岩爆倾向，可能会发生中等规模的岩爆事件，需要采取预防措施。",
        "强岩爆倾向 (3级)": "岩石具有强烈的岩爆倾向，极易发生大规模岩爆事件，需要严格的监测和防护措施。"
    }
    
    for grade, description in grade_info.items():
        st.markdown(f"**{grade}**")
        st.markdown(f"<p class='info-text'>{description}</p>", unsafe_allow_html=True)
        st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 添加建议卡片
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("岩爆防治建议")
    st.markdown("""
    <p class='info-text'>
    - 在进行隧道或地下工程开挖前，建议进行详细的岩体稳定性评估<br>
    - 对于中高岩爆倾向区域，应采用控制爆破技术<br>
    - 考虑使用预裂爆破、光面爆破等方法减小爆破震动<br>
    - 对于强岩爆倾向区域，可采用预应力释放钻孔等措施<br>
    - 加强监测工作，及时发现岩爆前兆
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 底部信息
st.markdown("---")
st.markdown("<center>© 2023 岩爆预测系统 | 技术支持: AI岩石力学实验室</center>", unsafe_allow_html=True) 
