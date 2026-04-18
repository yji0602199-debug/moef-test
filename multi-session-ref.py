"""
멀티세션 RAG 챗봇 — Supabase 세션/벡터 저장, 스트리밍 답변.
실행: streamlit run multi-session-ref.py (7.MultiService/code 디렉터리에서)
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from supabase import Client, create_client

# --- 경로: AI-Education 루트 ---
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _REPO_ROOT / ".env"
_LOG_DIR = _REPO_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- 로깅 (ref.txt: ERROR/WARNING만, HTTP 로그 억제) ---
for _name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(
            _LOG_DIR / f"chatbot_{datetime.now():%Y%m%d}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
_log = logging.getLogger("multi_session_ref")

# --- .env (절대경로) ---
load_dotenv(dotenv_path=_ENV_PATH, override=True)

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()

# LLM 모델명은 프롬프트 지시대로 고정
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536

LOGO_CANDIDATES = (
    _REPO_ROOT / "logo.png",
    _REPO_ROOT / "5.DatabaseSQL" / "code" / "logo.png",
)


def remove_separators(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"~~[^~]*~~", "", text)
    text = re.sub(r"^[\t ]*(---+|===+|___+)\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@st.cache_resource
def get_supabase() -> Client | None:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def get_llm(streaming: bool = True) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.7,
        api_key=OPENAI_API_KEY or None,
        streaming=streaming,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        dimensions=EMBED_DIMS,
        api_key=OPENAI_API_KEY or None,
    )


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY or None)


def fetch_sessions(sb: Client) -> list[dict[str, Any]]:
    res = (
        sb.table("chat_sessions")
        .select("id,title,created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return list(res.data or [])


def ensure_session_row(sb: Client, session_id: str, title: str = "임시 세션") -> None:
    chk = sb.table("chat_sessions").select("id").eq("id", session_id).execute()
    if chk.data:
        return
    sb.table("chat_sessions").insert({"id": session_id, "title": title}).execute()


def update_session_title(sb: Client, session_id: str, title: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    sb.table("chat_sessions").update({"title": title, "updated_at": now}).eq(
        "id", session_id
    ).execute()


def load_messages(sb: Client, session_id: str) -> list[dict[str, str]]:
    res = (
        sb.table("chat_messages")
        .select("role,content,created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )
    rows = res.data or []
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def sync_messages_to_db(sb: Client, session_id: str, messages: list[dict[str, str]]) -> None:
    sb.table("chat_messages").delete().eq("session_id", session_id).execute()
    if not messages:
        return
    batch: list[dict[str, Any]] = [
        {"session_id": session_id, "role": m["role"], "content": m["content"]}
        for m in messages
    ]
    for i in range(0, len(batch), 50):
        sb.table("chat_messages").insert(batch[i : i + 50]).execute()


def generate_session_title_from_qa(question: str, answer: str) -> str:
    try:
        oai = get_openai_client()
        r = oai.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": "첫 질문과 첫 답변을 바탕으로 세션 제목을 한 줄 한국어로 짧게(40자 이내)만 출력하세요. 따옴표나 접두어 없이 제목만.",
                },
                {
                    "role": "user",
                    "content": f"질문:\n{question[:2000]}\n\n답변:\n{answer[:2000]}",
                },
            ],
        )
        t = (r.choices[0].message.content or "").strip().splitlines()[0].strip()
        return t[:120] if t else "새 세션"
    except Exception as e:
        _log.warning("제목 생성 실패: %s", e)
        return "새 세션"


def generate_followup_questions(context: str) -> str:
    try:
        oai = get_openai_client()
        r = oai.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.5,
            messages=[
                {
                    "role": "system",
                    "content": "다음 대화 맥락에 이어서 사용자가 물어보면 좋은 후속 질문을 정확히 3개, 번호 목록으로 한국어로만 작성하세요.",
                },
                {"role": "user", "content": context[:6000]},
            ],
        )
        body = (r.choices[0].message.content or "").strip()
        return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{body}"
    except Exception as e:
        _log.warning("후속 질문 생성 실패: %s", e)
        return ""


def retrieve_by_rpc(
    sb: Client, session_id: str, query: str, embeddings: OpenAIEmbeddings, k: int = 10
) -> list[Document]:
    qvec = embeddings.embed_query(query)
    try:
        res = sb.rpc(
            "match_vector_documents",
            {
                "query_embedding": qvec,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        docs: list[Document] = []
        for row in res.data or []:
            meta = dict(row.get("metadata") or {})
            meta["file_name"] = row.get("file_name")
            meta["similarity"] = row.get("similarity")
            docs.append(Document(page_content=row.get("content") or "", metadata=meta))
        return docs
    except Exception as e:
        _log.warning("RPC 검색 실패, 빈 컨텍스트로 진행: %s", e)
        return []


def insert_vector_batch(sb: Client, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    sb.table("vector_documents").insert(rows).execute()


def process_pdfs_for_session(
    sb: Client,
    session_id: str,
    uploaded_files: list[Any],
    embeddings: OpenAIEmbeddings,
) -> list[str]:
    ensure_session_row(sb, session_id)
    tmp_root = Path(st.session_state.get("_tmp_dir") or tempfile.gettempdir()) / "ms_ref_pdf"
    tmp_root.mkdir(parents=True, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    processed_names: list[str] = []
    for uf in uploaded_files:
        name = uf.name
        processed_names.append(name)
        safe_name = re.sub(r"[^\w.\-]", "_", name)[:180]
        tmp = tmp_root / f"{uuid.uuid4().hex}_{safe_name}"
        tmp.write_bytes(uf.getvalue())
        try:
            loader = PyPDFLoader(str(tmp))
            pages = loader.load()
            for d in pages:
                d.metadata = dict(d.metadata or {})
                d.metadata["file_name"] = name
            chunks = splitter.split_documents(pages)
            for d in chunks:
                d.metadata["file_name"] = name
            texts = [d.page_content for d in chunks]
            for i in range(0, len(texts), 10):
                sub_docs = chunks[i : i + 10]
                sub_texts = texts[i : i + 10]
                vecs = embeddings.embed_documents(sub_texts)
                insert_rows = []
                for d, vec in zip(sub_docs, vecs, strict=True):
                    fn = d.metadata.get("file_name") or name
                    insert_rows.append(
                        {
                            "session_id": session_id,
                            "content": d.page_content,
                            "embedding": vec,
                            "file_name": fn,
                            "metadata": {
                                k: v
                                for k, v in (d.metadata or {}).items()
                                if isinstance(v, (str, int, float, bool))
                            },
                        }
                    )
                insert_vector_batch(sb, insert_rows)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return processed_names


def copy_vectors_between_sessions(sb: Client, from_id: str, to_id: str) -> None:
    res = (
        sb.table("vector_documents")
        .select("content,embedding,file_name,metadata")
        .eq("session_id", from_id)
        .execute()
    )
    rows = res.data or []
    for i in range(0, len(rows), 10):
        batch = []
        for r in rows[i : i + 10]:
            emb = r.get("embedding")
            if isinstance(emb, str):
                try:
                    emb = json.loads(emb)
                except json.JSONDecodeError:
                    continue
            batch.append(
                {
                    "session_id": to_id,
                    "content": r["content"],
                    "embedding": emb,
                    "file_name": r["file_name"] or "unknown.pdf",
                    "metadata": r.get("metadata") or {},
                }
            )
        insert_vector_batch(sb, batch)


def distinct_vector_filenames(sb: Client, session_id: str) -> list[str]:
    res = (
        sb.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    names = sorted({(r.get("file_name") or "").strip() for r in (res.data or []) if r.get("file_name")})
    return [n for n in names if n]


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "_tmp_dir" not in st.session_state:
        st.session_state._tmp_dir = tempfile.mkdtemp(prefix="multi_session_ref_")
    if "sess_pick_id" not in st.session_state:
        st.session_state.sess_pick_id = ""
    if "_session_title_done" not in st.session_state:
        st.session_state._session_title_done = False


def apply_loaded_session(sb: Client, sid: str) -> None:
    st.session_state.session_id = sid
    st.session_state.chat_history = load_messages(sb, sid)
    st.session_state.processed_files = distinct_vector_filenames(sb, sid)
    st.session_state._session_title_done = True


def autosave_session(sb: Client) -> None:
    sid = st.session_state.session_id
    msgs = st.session_state.chat_history
    if not msgs:
        return
    ensure_session_row(sb, sid)
    sync_messages_to_db(sb, sid, msgs)
    if st.session_state.get("_session_title_done"):
        return
    user_first = next((m["content"] for m in msgs if m["role"] == "user"), None)
    asst_first = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
    if user_first and asst_first:
        title = generate_session_title_from_qa(user_first, asst_first)
        update_session_title(sb, sid, title)
        st.session_state._session_title_done = True


def render_header() -> None:
    logo_path = next((p for p in LOGO_CANDIDATES if p.is_file()), None)
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if logo_path:
            st.image(str(logo_path), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
            <div style="text-align:center;">
                <span style="font-size:4rem !important; font-weight:800; color:#1f77b4;">멀티세션</span>
                <span style="font-size:4rem !important; font-weight:800; color:#ffd700;"> RAG 챗봇</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
        h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
        h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
        div.stButton > button:first-child {
            background-color: #ff69b4;
            color: #111;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_rag_messages(
    question: str, docs: list[Document], history: list[dict[str, str]]
) -> list[Any]:
    ctx = "\n\n".join(
        f"[{d.metadata.get('file_name', '문서')}] {d.page_content}" for d in docs[:10]
    )
    sys_txt = (
        "당신은 문서 기반 어시스턴트입니다. 제공된 컨텍스트와 이전 대화를 바탕으로 답하세요.\n"
        "답변은 # ## ### 헤딩으로 구조화하고, 존댓말·완전한 문장을 사용하세요.\n"
        "---, ===, ___, ~~취소선~~, 구분선은 사용하지 마세요. 참조/출처 문구는 넣지 마세요."
    )
    tail = history[-50:] if len(history) > 50 else history
    lc_msgs: list[Any] = [SystemMessage(content=sys_txt)]
    for m in tail:
        if m["role"] == "user":
            lc_msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_msgs.append(AIMessage(content=m["content"]))
    lc_msgs.append(
        HumanMessage(
            content=f"### 문서 컨텍스트\n{ctx}\n\n### 질문\n{question}"
        )
    )
    return lc_msgs


def main() -> None:
    st.set_page_config(
        page_title="멀티세션 RAG 챗봇",
        page_icon="📚",
        layout="wide",
    )
    inject_css()
    init_session_state()

    sb = get_supabase()
    missing: list[str] = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_ANON_KEY:
        missing.append("SUPABASE_ANON_KEY")

    render_header()

    if missing:
        st.error(
            "다음 환경 변수가 .env에 없습니다: "
            + ", ".join(missing)
            + f"\n\n기대 경로: `{_ENV_PATH}`"
        )
        return

    if sb is None:
        st.error("Supabase 클라이언트를 만들 수 없습니다.")
        return

    # --- 사이드바 (구분선 없음) ---
    st.sidebar.markdown("#### LLM 모델")
    st.sidebar.text(f"고정 모델: {LLM_MODEL}")

    sessions = fetch_sessions(sb)
    options_ids = [""] + [str(s["id"]) for s in sessions]
    id_labels: dict[str, str] = {"": "— 세션 선택 —"}
    for s in sessions:
        id_labels[str(s["id"])] = (s.get("title") or "제목 없음")[:80]

    if st.session_state.sess_pick_id and st.session_state.sess_pick_id not in options_ids:
        st.session_state.sess_pick_id = ""

    def _on_session_pick_change() -> None:
        sb2 = get_supabase()
        sel = st.session_state.get("sess_pick_id") or ""
        if not sb2 or not sel:
            return
        apply_loaded_session(sb2, sel)

    st.sidebar.markdown("#### 세션 선택")
    st.sidebar.selectbox(
        "세션 (선택 시 자동 로드)",
        options=options_ids,
        format_func=lambda x: id_labels.get(x, x),
        key="sess_pick_id",
        on_change=_on_session_pick_change,
    )
    chosen_id = (st.session_state.sess_pick_id or "").strip()

    if st.sidebar.button("세션저장"):
        msgs = st.session_state.chat_history
        if len(msgs) < 2:
            st.sidebar.warning("저장할 대화(질문·답변)가 충분하지 않습니다.")
        else:
            user_first = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst_first = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            new_id = str(uuid.uuid4())
            title = generate_session_title_from_qa(user_first, asst_first)
            ensure_session_row(sb, st.session_state.session_id)
            sync_messages_to_db(sb, st.session_state.session_id, msgs)
            sb.table("chat_sessions").insert({"id": new_id, "title": title}).execute()
            rows = [
                {"session_id": new_id, "role": m["role"], "content": m["content"]}
                for m in msgs
            ]
            for i in range(0, len(rows), 50):
                sb.table("chat_messages").insert(rows[i : i + 50]).execute()
            copy_vectors_between_sessions(sb, st.session_state.session_id, new_id)
            st.sidebar.success("새 세션으로 저장했습니다.")
            st.session_state.session_id = new_id
            st.session_state.sess_pick_id = new_id
            st.session_state._session_title_done = True
            st.rerun()

    if st.sidebar.button("세션로드"):
        if not chosen_id:
            st.sidebar.warning("풀다운에서 세션을 먼저 선택하세요.")
        else:
            apply_loaded_session(sb, chosen_id)
            st.sidebar.success("세션을 불러왔습니다.")
            st.rerun()

    if st.sidebar.button("세션삭제"):
        if not chosen_id:
            st.sidebar.warning("삭제할 세션을 선택하세요.")
        else:
            sb.table("chat_sessions").delete().eq("id", chosen_id).execute()
            if st.session_state.sess_pick_id == chosen_id:
                st.session_state.sess_pick_id = ""
            if st.session_state.session_id == chosen_id:
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.chat_history = []
                st.session_state.processed_files = []
                st.session_state._session_title_done = False
            st.sidebar.success("삭제했습니다.")
            st.rerun()

    if st.sidebar.button("화면초기화"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.processed_files = []
        st.session_state.sess_pick_id = ""
        st.session_state._session_title_done = False
        st.sidebar.success("화면을 초기화했습니다.")
        st.rerun()

    if st.sidebar.button("vectordb"):
        names = distinct_vector_filenames(sb, st.session_state.session_id)
        if names:
            st.sidebar.text("벡터 DB 파일명:\n" + "\n".join(names))
        else:
            st.sidebar.text("현재 세션에 저장된 벡터 문서가 없습니다.")

    st.sidebar.markdown("#### PDF 업로드")
    files = st.sidebar.file_uploader(
        "PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_up",
    )
    if st.sidebar.button("파일 처리하기"):
        if not files:
            st.sidebar.warning("PDF를 업로드하세요.")
        else:
            try:
                embs = get_embeddings()
                names = process_pdfs_for_session(
                    sb, st.session_state.session_id, list(files), embs
                )
                st.session_state.processed_files = sorted(
                    set(st.session_state.processed_files) | set(names)
                )
                autosave_session(sb)
                st.sidebar.success(f"처리 완료: {', '.join(names)}")
            except Exception as e:
                _log.error("PDF 처리 오류: %s", e, exc_info=True)
                st.sidebar.error(f"처리 중 오류: {e}")

    st.sidebar.markdown("#### 현재 설정")
    st.sidebar.text(
        f"모델: {LLM_MODEL}\n"
        f"세션 ID: {st.session_state.session_id[:8]}…\n"
        f"처리된 파일 수: {len(st.session_state.processed_files)}\n"
        f"대화 수: {len(st.session_state.chat_history)}"
    )

    # --- 채팅 표시 ---
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(remove_separators(m["content"]), unsafe_allow_html=False)

    prompt = st.chat_input("질문을 입력하세요")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(remove_separators(prompt))

    embeddings = get_embeddings()
    docs = retrieve_by_rpc(sb, st.session_state.session_id, prompt, embeddings, k=10)
    llm = get_llm(streaming=True)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        try:
            if docs:
                lc_in = build_rag_messages(prompt, docs, st.session_state.chat_history[:-1])
                for chunk in llm.stream(lc_in):
                    if chunk.content:
                        full += chunk.content
                        placeholder.markdown(remove_separators(full))
            else:
                sys_direct = (
                    "당신은 도움이 되는 어시스턴트입니다. "
                    "답변은 # ## ### 헤딩으로 구조화하고 존댓말을 사용하세요. "
                    "구분선, 취소선, 참조 문구는 사용하지 마세요."
                )
                msgs: list[Any] = [SystemMessage(content=sys_direct)]
                for m in st.session_state.chat_history[:-1][-50:]:
                    if m["role"] == "user":
                        msgs.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        msgs.append(AIMessage(content=m["content"]))
                msgs.append(HumanMessage(content=prompt))
                for chunk in llm.stream(msgs):
                    if chunk.content:
                        full += chunk.content
                        placeholder.markdown(remove_separators(full))

            ctx_for_follow = f"질문:\n{prompt}\n\n답변:\n{full}"
            follow = generate_followup_questions(ctx_for_follow)
            full_final = remove_separators(full + follow)
            placeholder.markdown(full_final)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": full_final}
            )
        except Exception as e:
            _log.error("응답 생성 오류: %s", e, exc_info=True)
            err = f"\n\n오류가 발생했습니다: {e}"
            placeholder.markdown(remove_separators(full + err))
            st.session_state.chat_history.append(
                {"role": "assistant", "content": remove_separators(full + err)}
            )

    try:
        autosave_session(sb)
    except Exception as e:
        _log.warning("자동 저장 실패: %s", e)


main()
