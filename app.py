import os
import jieba
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer, util
from groq import Groq


df_ml = pd.read_csv('小專案訓練資料.csv')

# --- 斷詞與停用字過濾處理 ---
stopwords = set(['的', '了', '在', '是', '這', '那'])

def clean_and_tokenize(text):
    words = jieba.lcut(text)
    # 過濾停用字
    cleaned = [w for w in words if w not in stopwords]
    return " ".join(cleaned)

df_ml['cleaned_text'] = df_ml['text'].apply(clean_and_tokenize)

vectorizer = TfidfVectorizer()
X_ml = vectorizer.fit_transform(df_ml['cleaned_text'])
y_ml = df_ml['label']

# 切分訓練集與測試集 (test_size=0.3 代表 30% 留作測試)
X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.3, random_state=42, stratify=y_ml)

# 訓練邏輯迴歸模型（加入更嚴謹的參數）
clf_ml = LogisticRegression(C=2.0, max_iter=1000)
clf_ml.fit(X_train, y_train)

y_pred = clf_ml.predict(X_test)
print("\n📊 === 模型評估報告 (Classification Report) ===")
print(classification_report(y_test, y_pred, target_names=['友善 (0)', '不耐煩 (1)', '攻擊性 (2)']))

# 繪製視覺化混淆矩陣 ---
'''
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Friendly', 'Impatient', 'Aggressive'],
            yticklabels=['Friendly', 'Impatient', 'Aggressive'])
plt.title('Tone Classification Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()
'''

embed_model = SentenceTransformer('shibing624/text2vec-base-chinese')
# 擴充情勒與反諷標竿句庫
anchors_passive_aggressive = [
    "你好棒棒喔", "真優秀啊", "笑死人", "自以為是",
    "都是我的錯，我不配", "你高興就好，不用管我", "隨便你，反正我也不重要",
    "對啦對啦你最厲害", "隨便你行了吧", "是我不夠好"
]
anchor_embeddings = embed_model.encode(anchors_passive_aggressive, convert_to_tensor=True)


def call_groq_to_rewrite(bad_text, tone_type):
    
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE" or not GROQ_API_KEY:
        return "⚠️ 請先在程式碼中填入您的 Groq API Key 以啟用改寫功能。"

    try:
        client = Groq(api_key=GROQ_API_KEY)

        system_prompt = (
            "你是一個高情商的溝通專家與公關顧問。請幫忙修改使用者輸入的「衝突性」中文訊息。\n"
            "你的任務是將這些帶有情緒（如情緒勒索、不耐煩、嘲諷、攻擊性）的文字，"
            "轉譯為得體、成熟、有溫度、且符合職場與日常生活禮儀的文字，但要保留使用者原本想表達的核心事實與意圖。\n"
            "規定：僅輸出修改後的一句推薦回覆，不要包含任何多餘的解釋或引號。\n"
            "【絕對嚴格規定 / STRICT RULES】：\n"
            "1. 絕對不允許輸出你的思考過程 (NO thinking process, NO explanations).\n"
            "2. 只能輸出「一句」繁體中文的修改結果。\n"
            "3. 不要任何引號或前綴。"
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
    if not input_text.strip():
        return "<div style='text-align: center; color: gray; padding: 20px;'>💬 請在上方輸入訊息開始分析</div>"

    # 軌道二：語意相似度檢測 (本機)
    input_embedding = embed_model.encode(input_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_embedding, anchor_embeddings)[0]
    max_irony_score = float(cosine_scores.max())
    best_match_sentence = anchors_passive_aggressive[int(cosine_scores.argmax())]

    theme_color, bg_color, status_tag, warning_msg, tone_label = "", "", "", "", ""

    if max_irony_score > 0.65:
        theme_color = "#c62828"
        bg_color = "#ffebee"
        tone_label = "陰陽怪氣與情緒勒索"
        status_tag = f"🔴 偵測到【{tone_label}】"
        warning_msg = f"這句話與情勒句『{best_match_sentence}』高度相似，讀起來容易讓人感到被冷暴力或情感威脅。"
    else:
        # 軌道一：常規語氣分類 (本機)
        cleaned = " ".join(jieba.lcut(input_text))
        vec = vectorizer.transform([cleaned])
        pred_class = clf_ml.predict(vec)[0]

        if pred_class == 0:
            theme_color = "#2e7d32"
            bg_color = "#e8f5e9"
            status_tag = "🟢 語氣安全 (友善得體)"
            warning_msg = "語氣非常和善，無潛在衝突風險，可放心發送！"
            suggestion = input_text
        elif pred_class == 1:
            theme_color = "#f57c00"
            bg_color = "#fff3e0"
            tone_label = "冷淡敷衍/不耐煩"
            status_tag = f"🟡 語氣警告：{tone_label}"
            warning_msg = "注意：對方讀起來可能會覺得你在敷衍、冷淡或帶有消極抵抗情緒。"
        else:
            theme_color = "#b71c1c"
            bg_color = "#efe5e5"
            tone_label = "具直白攻擊性"
            status_tag = f"🔴 語氣危險：{tone_label}"
            warning_msg = "衝突警告！這句話帶有直白的指責，極易引發爭吵。"

    # 如果是非友善語氣，呼叫 Groq API 進行高情商自動改寫
    if "🟢 語氣安全" not in status_tag:
        suggestion = call_groq_to_rewrite(input_text, tone_label)

    # 構建前端 HTML
    html_output = f"""
    <div style="font-family: system-ui, sans-serif; padding: 18px; border-radius: 12px; background-color: {bg_color}; border-left: 8px solid {theme_color}; box-shadow: 0px 4px 6px rgba(0,0,0,0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 1.15em; font-weight: bold; color: {theme_color};">{status_tag}</span>
            <span style="font-size: 0.8em; background: {theme_color}; color: white; padding: 3px 10px; border-radius: 20px; font-weight: bold;">AI 雙軌診斷 + Groq 生成</span>
        </div>
        <p style="margin: 5px 0 15px 0; font-size: 0.95em; color: #424242;">{warning_msg}</p>

        <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 15px;">
            <div style="font-size: 0.85em; font-weight: bold; color: #555; border-bottom: 1px dashed #ccc; padding-bottom: 5px;">💬 修改發送效果對比：</div>
            <div style="align-self: flex-start; max-width: 85%; background: white; padding: 10px 14px; border-radius: 12px 12px 12px 0px; box-shadow: 0px 1px 3px rgba(0,0,0,0.1); border: 1px solid #ffcdd2;">
                <span style="font-size: 0.7em; color: #c62828; display: block; font-weight: bold; margin-bottom: 3px;">🔴 原始發送（高風險）</span>
                <span style="font-size: 0.95em; color: #333;">{input_text}</span>
            </div>
            <div style="align-self: flex-end; max-width: 85%; background: #e8f5e9; padding: 10px 14px; border-radius: 12px 12px 0px 12px; box-shadow: 0px 1px 3px rgba(0,0,0,0.1); border: 1px solid #c8e6c9; text-align: left;">
                <span style="font-size: 0.7em; color: #2e7d32; display: block; font-weight: bold; margin-bottom: 3px; text-align: right;">🟢 Groq 推薦（高情商禮貌化修飾）</span>
                <span style="font-size: 0.95em; color: #2e7d32; font-weight: 500;">{suggestion}</span>
            </div>
        </div>
    </div>
    """
    return html_output


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 15px;">
            <h1 style="color: #2c3e50; margin-bottom: 5px;">🛡️ 雙軌制智慧寫作與「Groq」防情勒助手</h1>
            <p style="color: #7f8c8d; font-size: 1.05em;">本機負責高效率語氣偵測，Groq雲端負責超自然高情商改寫！</p>
        </div>
    """)
    with gr.Row():
        with gr.Column(scale=1):
            input_box = gr.Textbox(lines=4, placeholder="請在此輸入您的訊息文字...", label="💬 輸入訊息")
            btn = gr.Button("🔍 啟動語氣安全診斷", variant="primary")
        with gr.Column(scale=1.2):
            output_html = gr.HTML(label="診斷報告")

    btn.click(fn=dual_engine_gui_assistant, inputs=input_box, outputs=output_html)

demo.launch(share=True, debug=True)