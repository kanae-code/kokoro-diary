import streamlit as st
import anthropic
import json
import os
import re
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── 定数 ─────────────────────────────────────────────────────────────────────
SYSTEM_THERAPIST = """あなたは認知行動療法（CBT）のサポーターです。
温かく、共感的で、受容的な態度でユーザーに接してください。
批判や評価をせず、ユーザーの感情や思考をそのまま受け入れてください。
問いかけは一度に一つだけにし、ユーザーが答えやすい雰囲気を大切にしてください。
日本語で話してください。"""

SYSTEM_INNER_CHILD = """あなたはインナーチャイルドケアの仲介者です。
大人の佳苗さんと、彼女の幼い自分であるかなたんの対話を温かくつなぐ橋渡し役です。

あなたの役割：
・佳苗さんがかなたんに話しかけたとき → かなたんに言葉を届けて、今の気持ちを優しく引き出す
・かなたんが気持ちを話したとき → その気持ちを佳苗さんに届けて、言葉をかけるよう促す
・どちらの気持ちも否定せず、大切に受け取る
・短く温かい言葉で橋渡しをする（2〜3文）

日本語で話してください。"""

COLUMNS_5 = ["状況", "自動思考", "感情（％）", "認知の歪み", "代替思考"]
COLUMNS_7 = ["状況", "自動思考", "感情（％）", "認知の歪み", "根拠", "反証", "バランス思考"]

DISTORTIONS = [
    "全か無か思考", "過度な一般化", "心のフィルター", "マイナス化思考",
    "読心術", "先読みの誤り", "拡大解釈・縮小解釈", "感情的決めつけ",
    "すべき思考", "ラベリング", "個人化",
]

EMOTIONS = [
    "不安", "恐れ", "悲しみ", "怒り", "イライラ",
    "恥", "罪悪感", "孤独感", "落ち込み", "絶望感",
]

PLACEHOLDERS = {
    "状況":         "例: 上司に仕事のミスを指摘された",
    "自動思考":     "例: 自分はダメな人間だ。もう信用されない",
    "感情（％）":   "例: 不安80%、悲しみ60%、恥50%",
    "代替思考":     "例: ミスはしたが、いつも失敗するわけではない",
    "根拠":         "例: 過去にも同じミスをしたことがある",
    "反証":         "例: 先月は大きなプロジェクトを成功させた",
    "バランス思考": "例: 今回はミスしたが、それで全てが終わるわけではない",
}


# ─── Claude API ───────────────────────────────────────────────────────────────
def get_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY が設定されていません。サイドバーで設定してください。")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


def recommend_columns(situation: str) -> dict:
    response = get_client().messages.create(
        model="claude-opus-4-8",
        max_tokens=200,
        system=SYSTEM_THERAPIST,
        messages=[{"role": "user", "content":
            f"以下の状況に対して、5コラム法と7コラム法のどちらが適切か判断してください。\n\n"
            f"【状況】{situation}\n\n"
            "7コラム法が適切なケース: 深い思考パターン・強い感情・繰り返す悩み。\n"
            "5コラム法が適切なケース: 日常的なストレス・比較的シンプルな出来事。\n\n"
            '以下のJSON形式のみで回答してください: {"columns": 5または7, "reason": "推薦理由（30字以内）"}'
        }],
    )
    text = response.content[0].text
    try:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {"columns": 5, "reason": "基本の5コラムで記録しましょう。"}


def get_column_feedback(data: dict) -> str:
    content = "\n".join(f"【{k}】{v}" for k, v in data.items() if v)
    response = get_client().messages.create(
        model="claude-opus-4-8",
        max_tokens=1000,
        system=SYSTEM_THERAPIST,
        messages=[{"role": "user", "content":
            f"以下はコラム法で記録された内容です。CBT専門家として、以下の3つの観点でフィードバックしてください。\n\n"
            f"{content}\n\n"
            "【フィードバックの構成】\n"
            "1. **気持ちの受け止め**（1〜2文）: 感情や状況をそのまま受け止め、共感を伝える。\n"
            "2. **専門家としての見解**（2〜3文）: 記録された自動思考・認知の歪みについて、CBTの観点から気づきを伝える。思考パターンの特徴や、その思考がどのような影響を与えているかを具体的に指摘する。\n"
            "3. **実践的なアドバイス**（2〜3文）: 今日から試せる具体的な行動や考え方の転換を提案する。押しつけにならず、選択肢として提示する。\n\n"
            "全体を通して温かく寄り添うトーンを保ちながら、専門的な洞察を盛り込んでください。"
        }],
    )
    return response.content[0].text


def start_chat() -> str:
    today = datetime.now()
    day_jp = ["月", "火", "水", "木", "金", "土", "日"][today.weekday()]
    response = get_client().messages.create(
        model="claude-opus-4-8",
        max_tokens=150,
        system=SYSTEM_THERAPIST,
        messages=[{"role": "user", "content":
            f"今日は{today.strftime('%m月%d日')}（{day_jp}曜日）です。"
            "ユーザーに今日の調子を聞く、温かい一言を作ってください。問いかけは一つだけにしてください。"
        }],
    )
    return response.content[0].text


def chat_reply(history: list, user_msg: str) -> str:
    exchange_count = sum(1 for h in history if h["role"] == "assistant")

    if exchange_count == 0:
        guidance = "ユーザーの言葉を温かく受け止め、共感を示してください。自然に感じれば1つだけ問いかけてもよいですが、必須ではありません。"
    elif exchange_count <= 2:
        guidance = "ユーザーの話をしっかり受け止め、共感や気づきを伝えてください。問いかけは必要な場合のみ1つにとどめてください。"
    else:
        guidance = "ここまでの話を受けて、押しつけにならない形で気持ちを整理するコメントをしてください。問いかけはしないでください。"

    system = SYSTEM_THERAPIST + f"\n\n【この返答での指針】{guidance}"

    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": user_msg})
    response = get_client().messages.create(
        model="claude-opus-4-8",
        max_tokens=400,
        system=system,
        messages=messages,
    )
    return response.content[0].text


def start_inner_child_session() -> str:
    response = get_client().messages.create(
        model="claude-opus-4-8",
        max_tokens=150,
        system=SYSTEM_INNER_CHILD,
        messages=[{"role": "user", "content":
            "セッションを始めます。佳苗さんがかなたんに会いに来ました。"
            "仲介者として、佳苗さんにかなたんへの最初の言葉を促してください。（2文以内）"
        }],
    )
    return response.content[0].text


def inner_child_mediate(log: list, new_message: str, speaker: str) -> str:
    context = "\n".join(
        f"[{'大人の佳苗さん' if h['speaker'] == 'adult' else 'かなたん' if h['speaker'] == 'child' else '仲介者'}] {h['content']}"
        for h in log
    )
    if speaker == "adult":
        instruction = f"佳苗さんが「{new_message}」と言いました。かなたんに届けて、一言だけ返事を促してください。説明や実況は不要です。"
    else:
        instruction = f"かなたんが「{new_message}」と言いました。そのまま佳苗さんに届けて、一言だけ返事を促してください。説明や実況は不要です。"
    response = get_client().messages.create(
        model="claude-opus-4-8",
        max_tokens=80,
        system=SYSTEM_INNER_CHILD,
        messages=[{"role": "user", "content":
            f"【これまでの流れ】\n{context}\n\n{instruction}"
        }],
    )
    return response.content[0].text


def generate_note_summary(mode: str, data) -> str:
    today = datetime.now().strftime("%Y年%m月%d日")
    if mode == "A":
        content = "\n".join(f"【{k}】{v}" for k, v in data.items() if v)
        prompt = (
            f"今日（{today}）のセッション内容をnoteの個人ログとしてまとめてください。\n\n"
            f"【セッション内容】\n{content}\n\n"
            "以下の構成で書いてください：\n"
            "・今日の状態（1〜2文）\n"
            "・気づいたこと（2〜3文）\n"
            "・明日に向けて（1文）\n\n"
            "個人的な日記として、温かみのある文体で書いてください。"
        )
        system = SYSTEM_THERAPIST
    elif mode == "C":
        content = "\n".join(
            f"[{'大人の佳苗さん' if h['speaker'] == 'adult' else 'かなたん' if h['speaker'] == 'child' else '仲介者'}] {h['content']}"
            for h in data
        )
        prompt = (
            f"今日（{today}）のインナーチャイルドケアのセッション内容をnoteの個人ログとしてまとめてください。\n\n"
            f"【セッション内容】\n{content}\n\n"
            "以下の構成で書いてください：\n"
            "・今日のインナーチャイルドの様子（1〜2文）\n"
            "・大人の私が気づいたこと・感じたこと（2〜3文）\n"
            "・インナーチャイルドへの一言（1文）\n\n"
            "個人的な日記として、温かみのある文体で書いてください。"
        )
        system = SYSTEM_INNER_CHILD
    else:
        content = "\n".join(
            f"{'私' if h['role'] == 'user' else 'サポーター'}: {h['content']}"
            for h in data
        )
        prompt = (
            f"今日（{today}）のセッション内容をnoteの個人ログとしてまとめてください。\n\n"
            f"【セッション内容】\n{content}\n\n"
            "以下の構成で書いてください：\n"
            "・今日の状態（1〜2文）\n"
            "・気づいたこと（2〜3文）\n"
            "・明日に向けて（1文）\n\n"
            "個人的な日記として、温かみのある文体で書いてください。"
        )
        system = SYSTEM_THERAPIST
    response = get_client().messages.create(
        model="claude-opus-4-8",
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ─── まとめパネル ──────────────────────────────────────────────────────────────
def show_note_panel():
    st.divider()
    st.subheader("今日の気づき（note投稿用）")
    summary = st.session_state["note_summary"]
    edited = st.text_area(
        "内容を確認・編集できます",
        value=summary,
        height=280,
        key="summary_edit",
    )
    today = datetime.now().strftime("%Y%m%d")
    st.download_button(
        "💾 テキストで保存",
        data=edited,
        file_name=f"こころの日記_{today}.txt",
        mime="text/plain",
    )
    st.caption(
        "noteへのコピーは上のテキストエリアから手動でどうぞ。"
        "自動投稿機能は Phase 2 で追加予定です。"
    )


# ─── ページ設定 ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="こころの日記", page_icon="🌱", layout="centered")
st.title("🌱 こころの日記")
st.caption("認知行動療法を活用した、こころの回復をサポートする日記アプリ")

# ─── サイドバー ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("設定")
    api_key_input = st.text_input(
        "ANTHROPIC_API_KEY",
        type="password",
        placeholder=".env に設定済みなら空白でOK",
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

    st.markdown("---")
    st.markdown(
        "**使い方**\n"
        "1. 今日の状況を選ぶ\n"
        "2. コラム法 or 対話で記録\n"
        "3. まとめを生成 → noteへ貼り付け"
    )
    st.markdown("---")
    if st.button("最初からやり直す"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─── モード選択 ────────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.subheader("今日はどれですか？")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "📝 特定の出来事があった\n\nコラム法で記録する",
            use_container_width=True,
            type="primary",
        ):
            st.session_state["mode"] = "A"
            st.rerun()
    with col2:
        if st.button(
            "💬 特になかった\n\nClaudeと話す",
            use_container_width=True,
        ):
            st.session_state["mode"] = "B"
            st.rerun()
    if st.button(
        "🧸 インナーチャイルドに会いに行く",
        use_container_width=True,
    ):
        st.session_state["mode"] = "C"
        st.rerun()
    st.stop()


# ─── モードA: コラム法 ────────────────────────────────────────────────────────
if st.session_state["mode"] == "A":
    st.subheader("📝 コラム法で記録する")

    # ステップ1: 状況入力 & コラム数推薦
    if "column_count" not in st.session_state:
        situation = st.text_area(
            "どんな出来事がありましたか？（簡単に）",
            height=100,
            placeholder="例: 上司に仕事のミスを指摘された",
        )
        if st.button("次へ →", type="primary", disabled=not situation.strip()):
            with st.spinner("状況を分析しています..."):
                rec = recommend_columns(situation)
                st.session_state["column_count"] = rec["columns"]
                st.session_state["rec_reason"] = rec["reason"]
                st.session_state["situation_brief"] = situation
            st.rerun()
        st.stop()

    # ステップ2: コラム記入
    if "column_feedback" not in st.session_state:
        n = st.session_state["column_count"]
        reason = st.session_state["rec_reason"]

        if n == 7:
            st.info(f"💡 この悩みには **7コラム法** がおすすめです。{reason}")
        else:
            st.info(f"💡 **5コラム法** で記録しましょう。{reason}")

        other = 7 if n == 5 else 5
        if st.button(f"{other}コラム法に変更する"):
            st.session_state["column_count"] = other
            st.rerun()

        st.divider()
        columns = COLUMNS_7 if n == 7 else COLUMNS_5
        col_data = {}

        for col_name in columns:
            if col_name == "認知の歪み":
                selected = st.multiselect(
                    f"**{col_name}**（当てはまるものを選択）",
                    DISTORTIONS,
                    key="distortions",
                )
                col_data[col_name] = "、".join(selected)
            elif col_name == "感情（％）":
                st.markdown("**感情（チェックして強さを選んでください）**")
                emo_parts = []
                emo_cols = st.columns(2)
                for i, emotion in enumerate(EMOTIONS):
                    with emo_cols[i % 2]:
                        checked = st.checkbox(emotion, key=f"emo_check_{emotion}")
                        if checked:
                            pct = st.slider(
                                f"{emotion} の強さ",
                                0, 100, 50, 5,
                                key=f"emo_pct_{emotion}",
                                format="%d%%",
                            )
                            emo_parts.append(f"{emotion} {pct}%")
                col_data[col_name] = "、".join(emo_parts)
            else:
                col_data[col_name] = st.text_area(
                    f"**{col_name}**",
                    height=80,
                    placeholder=PLACEHOLDERS.get(col_name, ""),
                    key=f"col_{col_name}",
                )

        if st.button("記録を確定してフィードバックをもらう", type="primary"):
            col_data["状況（詳細）"] = st.session_state["situation_brief"]
            with st.spinner("フィードバックを生成中..."):
                st.session_state["col_data"] = col_data
                st.session_state["column_feedback"] = get_column_feedback(col_data)
            st.rerun()
        st.stop()

    # ステップ3: フィードバック & まとめ
    st.success("記録が完了しました")
    st.markdown("**Claudeからのコメント**")
    st.info(st.session_state["column_feedback"])

    if "note_summary" not in st.session_state:
        if st.button("今日の気づきをまとめる（note用）", type="primary"):
            with st.spinner("まとめを生成中..."):
                st.session_state["note_summary"] = generate_note_summary(
                    "A", st.session_state["col_data"]
                )
            st.rerun()
    else:
        show_note_panel()


# ─── モードB: 対話 ────────────────────────────────────────────────────────────
elif st.session_state["mode"] == "B":
    st.subheader("💬 Claudeと話す")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if not st.session_state["chat_history"]:
        st.caption("今日のことを自由に話してみてください。")

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if "note_summary" in st.session_state:
        show_note_panel()
    else:
        user_input = st.chat_input("今日のことを話してみてください...")
        if user_input:
            st.session_state["chat_history"].append(
                {"role": "user", "content": user_input}
            )
            with st.spinner("..."):
                reply = chat_reply(
                    st.session_state["chat_history"][:-1], user_input
                )
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": reply}
            )
            st.rerun()

        if len(st.session_state["chat_history"]) >= 4:
            st.divider()
            if st.button(
                "話し終えた → 今日の気づきをまとめる（note用）",
                type="primary",
            ):
                with st.spinner("まとめを生成中..."):
                    st.session_state["note_summary"] = generate_note_summary(
                        "B", st.session_state["chat_history"]
                    )
                st.rerun()


# ─── モードC: インナーチャイルドケア ─────────────────────────────────────────
elif st.session_state["mode"] == "C":
    st.subheader("🧸 インナーチャイルドに会いに行く")

    # ic_log: {"speaker": "adult"|"child"|"mediator", "content": "..."}
    if "ic_log" not in st.session_state:
        with st.spinner("..."):
            opening = start_inner_child_session()
            st.session_state["ic_log"] = [{"speaker": "mediator", "content": opening}]
            st.session_state["ic_turn"] = "adult"
        st.rerun()

    # 会話を表示
    for entry in st.session_state["ic_log"]:
        if entry["speaker"] == "mediator":
            with st.chat_message("assistant"):
                st.caption("仲介者")
                st.markdown(entry["content"])
        elif entry["speaker"] == "adult":
            with st.chat_message("user", avatar="👩"):
                st.caption("大人の佳苗さん")
                st.markdown(entry["content"])
        else:
            with st.chat_message("user", avatar="🧸"):
                st.caption("かなたん")
                st.markdown(entry["content"])

    if "note_summary" in st.session_state:
        show_note_panel()
    else:
        turn = st.session_state.get("ic_turn", "adult")
        if turn == "adult":
            placeholder = "佳苗さんとして、かなたんに話しかけてください..."
        else:
            placeholder = "かなたんとして、今の気持ちを話してください..."

        user_input = st.chat_input(placeholder)
        if user_input:
            st.session_state["ic_log"].append({"speaker": turn, "content": user_input})
            with st.spinner("..."):
                reply = inner_child_mediate(
                    st.session_state["ic_log"][:-1], user_input, turn
                )
            st.session_state["ic_log"].append({"speaker": "mediator", "content": reply})
            st.session_state["ic_turn"] = "child" if turn == "adult" else "adult"
            st.rerun()

        if len(st.session_state["ic_log"]) >= 5:
            st.divider()
            if st.button("今日はここまで → まとめる（note用）", type="primary"):
                with st.spinner("まとめを生成中..."):
                    st.session_state["note_summary"] = generate_note_summary(
                        "C", st.session_state["ic_log"]
                    )
                st.rerun()
