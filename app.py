import os
import joblib
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from groq import Groq

# 🌟 設定 Streamlit 網頁標題與寬度
st.set_page_config(page_title="防情勒助手", layout="wide")

# 🌟 使用快取 (Cache) 載入模型，避免每次按按鈕都重新讀取
@st.cache_resource
def load_models():
    try:
        clf_ml = joblib.load('logistic_model.pkl')
        anchors_data = joblib.load('anchors_data.pkl')
        anchors_passive_aggressive = anchors_data['texts']
        anchor_embeddings = anchors_data['embeddings']
    except FileNotFoundError:
        st.error("❌ 找不到 logistic_model.pkl 或 anchors_data.pkl，請確保檔案已上傳至 GitHub。")
        st.stop()
    
    # 載入語意模型
    embed = SentenceTransformer('shibing624/text2vec-base-chinese')
    return clf, embed

clf_ml, embed_model = load_models()

@st.cache_resource

def call_groq_to_rewrite(bad_text, tone_type):
    # 改從 Streamlit Secrets 或 環境變數讀取 API Key，避免外洩
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    if not GROQ_API_KEY:
        return "⚠️ 系統尚未設定 Groq API Key，無法啟用改寫功能。"
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system_prompt = (
            "你是一個高情商的溝通專家與公關顧問。請幫忙修改使用者輸入的「衝突性」中文訊息。\n"
            "你的任務是將這些帶有情緒的文字，轉譯為得體、成熟、有溫度的文字，但保留核心意圖。\n"
            "規定：僅輸出修改後的一句推薦回覆，不要包含任何解釋或引號。"
        )
        user_prompt = f"原句：『{bad_text}』\n偵測到的語氣：{tone_type}\n請給出改寫建議："
        
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=256
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Groq API 呼叫失敗：{str(e)}"

def dual_engine_gui_assistant(input_text):
    # 軌道二：語意相似度檢測 (本機)
    input_embedding = embed_model.encode(input_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_embedding, anchor_embeddings)[0]
    max_irony_score = float(cosine_scores.max())
    best_match_sentence = anchors_passive_aggressive[int(cosine_scores.argmax())]
    
    theme_color, bg_color, status_tag, warning_msg, tone_label = "", "", "", "", ""
    
    if max_irony_score > 0.65:
        theme_color, bg_color = "#c62828", "#ffebee"
        tone_label = "陰陽怪氣與情緒勒索"
        status_tag = f"🔴 偵測到【{tone_label}】"
        warning_msg = f"這句話與情勒句『{best_match_sentence}』高度相似，讀起來容易讓人感到被冷暴力或情感威脅。"
    else:
        # 軌道一：常規語氣分類 (本機)
        vec = embed_model.encode([input_text])
        pred_class = clf_ml.predict(vec)[0]
        
        if pred_class == 0:
            theme_color, bg_color = "#2e7d32", "#e8f5e9"
            status_tag = "🟢 語氣安全 (友善得體)"
            warning_msg = "語氣非常和善，無潛在衝突風險，可放心發送！"
            suggestion = input_text
        elif pred_class == 1:
            theme_color, bg_color = "#f57c00", "#fff3e0"
            tone_label = "冷淡敷衍/不耐煩"
            status_tag = f"🟡 語氣警告：{tone_label}"
            warning_msg = "注意：對方讀起來可能會覺得你在敷衍、冷淡或帶有消極抵抗情緒。"
        else:
            theme_color, bg_color = "#b71c1c", "#efe5e5"
            tone_label = "具直白攻擊性"
            status_tag = f"🔴 語氣危險：{tone_label}"
            warning_msg = "衝突警告！這句話帶有直白的指責，極易引發爭吵。"
            
    if "🟢 語氣安全" not in status_tag:
        suggestion = call_groq_to_rewrite(input_text, tone_label)
        
    html_output = f"""
    <div style="font-family: system-ui, sans-serif; padding: 18px; border-radius: 12px; background-color: {bg_color}; border-left: 8px solid {theme_color}; box-shadow: 0px 4px 6px rgba(0,0,0,0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 1.15em; font-weight: bold; color: {theme_color};">{status_tag}</span>
            <span style="font-size: 0.8em; background: {theme_color}; color: white; padding: 3px 10px; border-radius: 20px; font-weight: bold;">AI 雙軌診斷 + Groq 生成</span>
        </div>
        <p style="margin: 5px 0 15px 0; font-size: 0.95em; color: #424242;">{warning_msg}</p>
        <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 15px;">
            <div style="font-size: 0.85em; font-weight: bold; color: #555; border-bottom: 1px dashed #ccc; padding-bottom: 5px;">💬 修改發送效果對比：</div>
            <div style="align-self: flex-start; max-width: 85%; background: white; padding: 10px 14px; border-radius: 12px 12px 12px 0px; border: 1px solid #ffcdd2;">
                <span style="font-size: 0.7em; color: #c62828; display: block; font-weight: bold; margin-bottom: 3px;">🔴 原始發送</span>
                <span style="font-size: 0.95em; color: #333;">{input_text}</span>
            </div>
            <div style="align-self: flex-end; max-width: 85%; background: #e8f5e9; padding: 10px 14px; border-radius: 12px 12px 0px 12px; border: 1px solid #c8e6c9; text-align: left;">
                <span style="font-size: 0.7em; color: #2e7d32; display: block; font-weight: bold; margin-bottom: 3px; text-align: right;">🟢 Groq 推薦（禮貌化修飾）</span>
                <span style="font-size: 0.95em; color: #2e7d32; font-weight: 500;">{suggestion}</span>
            </div>
        </div>
    </div>
    """
    return html_output

# 🌟 Streamlit 前端介面排版
st.title("🛡️ 智慧改寫與防情勒助手")
st.markdown("本機負責語氣偵測，Groq 自然改寫！")

# 將畫面分為左右兩欄 (比例 1 : 1.2)
col1, col2 = st.columns([1, 1.2])

with col1:
    input_box = st.text_area("💬 輸入訊息", placeholder="請在此輸入您的訊息文字...", height=150)
    # 按下按鈕後，submit_btn 會變成 True
    submit_btn = st.button("🔍 啟動語氣安全診斷", type="primary")

with col2:
    st.markdown("##### 診斷報告")
    if submit_btn:
        if input_box.strip():
            with st.spinner("分析與改寫中，請稍候..."):
                result = dual_engine_gui_assistant(input_box)
                st.markdown(result, unsafe_allow_html=True)
        else:
            st.info("💬 請在左側輸入訊息開始分析")
