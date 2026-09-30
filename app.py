"""
AI-Powered Visual Data Analytics and Business Intelligence Platform - Streamlit dashboard (MILESTONE 4)
--------------------------------------------------
Run:  streamlit run app.py

Pages:
  🏠 Home          overview + status
  👷 PPE Detection image / video          (Milestone 1)
  📄 Documents     upload + search manuals (Milestone 2)
  🤖 Ask AI        LangGraph agent          (Milestone 3)
  📊 Dashboard     KPIs + charts            (Milestone 4)
  📋 Report        PDF download             (Milestone 4)

Remember: Streamlit re-runs this whole file on every click.
  - @st.cache_resource  -> load heavy things only once
  - st.session_state    -> remember results between clicks
  - save data only inside a button, or it is saved again on every rerun
"""
import os
import tempfile

import cv2
import streamlit as st

import knowledge_base as kb
import storage
import vision
from agent import ask_agent
from documents import process_file
from llm import MODEL, has_key
from report import build_pdf, write_summary

st.set_page_config(page_title="AI-Powered Visual Data Analytics and Business Intelligence Platform", page_icon="🦺", layout="wide")

# ============================================================
# PROFESSIONAL UI / UX STYLING
# Frontend only - does not change backend functionality
# ============================================================

st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f6f8fc;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    h1, h2, h3 {
        color: #172033;
        font-weight: 700;
    }

    p {
        color: #526071;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111827 0%,
            #172554 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] h3 {
        color: white;
        font-size: 1.05rem;
        line-height: 1.5;
        padding-bottom: 12px;
    }


    /* ---------- NAVIGATION ---------- */

    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 9px 12px;
        margin: 4px 0;
        transition: all 0.2s ease;
    }

    div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.13);
    }


    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 9px;
        border: none;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(37,99,235,0.18);
    }


    /* ---------- METRIC CARDS ---------- */

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(15,23,42,0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #172033;
        font-weight: 750;
    }


    /* ---------- FILE UPLOADER ---------- */

    section[data-testid="stFileUploaderDropzone"] {
        background: #ffffff;
        border: 1.5px dashed #94a3b8;
        border-radius: 14px;
        padding: 12px;
    }

    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #2563eb;
        background: #f8fbff;
    }


    /* ---------- INPUTS ---------- */

    div[data-baseweb="input"] {
        border-radius: 10px;
    }

    div[data-baseweb="textarea"] {
        border-radius: 10px;
    }


    /* ---------- CARDS ---------- */

    .vd-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 5px 18px rgba(15,23,42,0.05);
    }

    .vd-card h3 {
        margin-top: 0;
        color: #172033;
    }

    .vd-card p {
        color: #64748b;
    }


    /* ---------- HERO ---------- */

    .vd-hero {
        background: linear-gradient(
            135deg,
            #172554 0%,
            #1d4ed8 55%,
            #2563eb 100%
        );

        padding: 38px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 12px 30px rgba(30,64,175,0.18);
    }

    .vd-hero h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 8px;
    }

    .vd-hero p {
        color: #dbeafe;
        font-size: 1.05rem;
    }


    /* ---------- STATUS BADGE ---------- */

    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: #dcfce7;
        color: #166534;
        font-weight: 600;
        font-size: 0.85rem;
    }


    /* ---------- SECTION TITLE ---------- */

    .section-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #172033;
        margin-top: 10px;
        margin-bottom: 18px;
    }


    /* ---------- FOOTER ---------- */

    .vd-footer {
        margin-top: 45px;
        padding: 20px;
        text-align: center;
        color: #64748b;
        border-top: 1px solid #e5e7eb;
    }

</style>
""", unsafe_allow_html=True)

# On Streamlit Cloud the API key comes from "Secrets"
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass


# ── load once ──────────────────────────────────────────────
@st.cache_resource
def get_model():
    return vision.load_model()


@st.cache_resource
def load_sample_manual():
    """If the knowledge base is empty (e.g. fresh cloud start), add a manual automatically."""
    if kb.total_chunks() == 0:
        for name in ["safety_manual.pdf", "sample_safety_manual.txt"]:
            path = os.path.join(vision.BASE_DIR, "data", name)
            if os.path.exists(path):
                kb.add_document(process_file(path), name)
                return name
    return None


def save_upload(uploaded):
    suffix = os.path.splitext(uploaded.name)[1]          # keep the real extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


model = get_model()
load_sample_manual()

# ── sidebar ────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div style="
        padding: 10px 0 20px 0;
        text-align: center;
    ">
        <div style="font-size: 2.4rem;">🛡️</div>
        <div style="
            font-size: 1.35rem;
            font-weight: 700;
            color: white;
        ">
            VisionDesk AI
        </div>
        <div style="
            font-size: 0.78rem;
            color: #94a3b8;
            margin-top: 5px;
        ">
            Workplace Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Home",
            "👷 PPE Detection",
            "📄 Documents",
            "🤖 Ask AI",
            "📊 Dashboard",
            "📋 Report"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### System Status")

    st.write(
        ("🟢" if model else "🔴") +
        " PPE Detection Model"
    )

    st.write(
        ("🟢" if has_key() else "🟡") +
        f" Gemini AI ({MODEL})"
    )

    st.write(
        f"📚 {kb.total_chunks()} knowledge chunks"
    )

    st.markdown("""
    <div style="
        margin-top: 30px;
        padding: 12px;
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        text-align: center;
        font-size: 0.75rem;
        color: #94a3b8;
    ">
        VisionDesk AI<br>
        Multimodal Safety Intelligence
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 🏠 HOME
# ══════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🛡️ VisionDesk AI")

    st.caption("Multimodal Workplace Intelligence Platform")

    st.success("🟢 AI SYSTEM ONLINE")

    st.markdown(
        """
        ### Workplace Intelligence, Simplified

        Detect safety risks, understand workplace documents,
        ask AI-powered questions, and generate compliance reports.
        """
    )

    st.divider()
    st.markdown(
        '<div class="section-title">Platform Capabilities</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="vd-card">
            <h3>👷 PPE Detection</h3>
            <p>
                Detect workers, protective equipment and
                PPE violations using computer vision.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="vd-card">
            <h3>📄 Document Intelligence</h3>
            <p>
                Upload safety manuals and retrieve
                relevant information using semantic search.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="vd-card">
            <h3>🤖 Safety AI</h3>
            <p>
                Ask workplace safety questions using
                documents and visual inspection results.
            </p>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="vd-card">
            <h3>📊 Business Intelligence</h3>
            <p>
                Monitor compliance rates, violations and
                safety trends through an interactive dashboard.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="vd-card">
            <h3>📋 Compliance Reports</h3>
            <p>
                Generate downloadable PDF reports containing
                workplace safety inspection results.
            </p>
        </div>
        """, unsafe_allow_html=True)
    if not model:
        st.warning("Put your trained **best.pt** in the project folder to enable detection.")
    if not has_key():
        st.info("Add **GEMINI_API_KEY** to the `.env` file for AI answers.")


# ══════════════════════════════════════════════════════════
# 👷 PPE DETECTION  (Milestone 1)
# ══════════════════════════════════════════════════════════
elif page == "👷 PPE Detection":

    # ────────────────────────────────────────────────────────
    # PROFESSIONAL PPE DETECTION UI
    # ────────────────────────────────────────────────────────
    st.markdown("""
    <style>

    .ppe-hero {
        background: linear-gradient(135deg, #172554 0%, #2563eb 100%);
        padding: 32px 36px;
        border-radius: 22px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.20);
    }

    .ppe-hero h1 {
        color: white;
        font-size: 2.4rem;
        margin-bottom: 8px;
        font-weight: 750;
    }

    .ppe-hero p {
        color: #dbeafe;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .status-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 22px;
        margin: 15px 0;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
    }

    .section-label {
        font-size: 1.05rem;
        font-weight: 700;
        color: #172554;
        margin-bottom: 8px;
    }

    .upload-card {
        background: white;
        border: 2px dashed #93c5fd;
        border-radius: 18px;
        padding: 20px;
        margin-top: 10px;
    }

    .result-card {
        background: white;
        border-radius: 18px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
        margin-top: 25px;
    }

    .metric-title {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 750;
        color: #172554;
    }

    .compliant-card {
        background: #ecfdf5;
        border: 1px solid #86efac;
        color: #166534;
        border-radius: 14px;
        padding: 18px;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 18px;
    }

    .violation-card {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        color: #991b1b;
        border-radius: 14px;
        padding: 18px;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 18px;
    }

    .info-card {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        border-radius: 14px;
        padding: 15px;
        margin-top: 15px;
    }

    </style>
    """, unsafe_allow_html=True)


    # ────────────────────────────────────────────────────────
    # HERO SECTION
    # ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="ppe-hero">
        <h1>👷 PPE Safety Inspection</h1>
        <p>
            AI-powered workplace safety monitoring using computer vision
            to identify protective equipment and potential PPE violations.
        </p>
    </div>
    """, unsafe_allow_html=True)


    # ────────────────────────────────────────────────────────
    # MODEL CHECK
    # ────────────────────────────────────────────────────────
    if not model:
        st.error("best.pt not found in the project folder.")
        st.stop()


    # ────────────────────────────────────────────────────────
    # INPUT SETTINGS
    # ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Inspection Settings</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        kind = st.radio(
            "Inspection type",
            ["Image", "Video"],
            horizontal=True
        )

    with col2:
        conf = st.slider(
            "Detection confidence",
            0.1,
            0.9,
            0.4,
            0.05,
            help="Higher confidence reduces false detections but may miss some objects."
        )

    st.markdown("<br>", unsafe_allow_html=True)


    # ────────────────────────────────────────────────────────
    # IMAGE MODE
    # ────────────────────────────────────────────────────────
    if kind == "Image":

        st.markdown("""
        <div class="status-card">
            <div class="section-label">📷 Upload Workplace Image</div>
            <div style="color:#64748b;">
                Upload a workplace image to detect workers,
                PPE equipment and safety violations.
            </div>
        </div>
        """, unsafe_allow_html=True)

        file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG and PNG"
        )

        if file:
            st.success(f"Selected: {file.name}")

        analyze = st.button(
            "🔍 Analyze Workplace Image",
            type="primary",
            use_container_width=True
        )

        if file and analyze:

            with st.spinner("AI is analyzing the workplace image..."):

                path = save_upload(file)

                # Existing backend logic — unchanged
                dets = vision.detect(model, path, conf)

                result = vision.summarize(
                    dets,
                    file.name
                )

                storage.log_scan(
                    file.name,
                    result,
                    "image"
                )

                st.session_state["last"] = {
                    "name": file.name,
                    "result": result,
                    "dets": dets,
                    "image": vision.draw(
                        cv2.imread(path),
                        dets
                    )
                }

            st.success("Inspection completed successfully.")


    # ────────────────────────────────────────────────────────
    # VIDEO MODE
    # ────────────────────────────────────────────────────────
    else:

        st.markdown("""
        <div class="status-card">
            <div class="section-label">🎥 Upload Workplace Video</div>
            <div style="color:#64748b;">
                Analyze workplace video frames to identify
                PPE equipment and safety violations.
            </div>
        </div>
        """, unsafe_allow_html=True)

        file = st.file_uploader(
            "Choose a video",
            type=["mp4", "avi", "mov"],
            help="Supported formats: MP4, AVI and MOV"
        )

        every_n = st.slider(
            "Frame sampling interval",
            1,
            30,
            5,
            help="Analyze one frame for every N frames."
        )

        if file:
            st.success(f"Selected: {file.name}")

        analyze_video = st.button(
            "🎥 Analyze Workplace Video",
            type="primary",
            use_container_width=True
        )

        if file and analyze_video:

            with st.spinner("AI is analyzing the workplace video..."):

                # Existing backend logic — unchanged
                dets, preview, frames = vision.detect_video(
                    model,
                    save_upload(file),
                    every_n,
                    conf
                )

                result = vision.summarize(
                    dets,
                    file.name
                )

                storage.log_scan(
                    file.name,
                    result,
                    "video"
                )

                st.session_state["last"] = {
                    "name": f"{file.name} ({frames} frames)",
                    "result": result,
                    "dets": dets,
                    "image": preview
                }

            st.success("Video inspection completed successfully.")


    # ────────────────────────────────────────────────────────
    # RESULTS
    # ────────────────────────────────────────────────────────
    last = st.session_state.get("last")

    if last:

        st.markdown("""
        <div class="section-label">
            📊 Inspection Results
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])

        # LEFT — IMAGE
        with col1:

            if last["image"] is not None:

                st.image(
                    cv2.cvtColor(
                        last["image"],
                        cv2.COLOR_BGR2RGB
                    ),
                    caption=last["name"],
                    use_container_width=True
                )


        # RIGHT — RESULTS
        with col2:

            r = last["result"]

            if r["compliant"]:

                st.markdown("""
                <div class="compliant-card">
                    🟢 COMPLIANT<br>
                    <span style="font-weight:500;">
                    No PPE violations detected.
                    </span>
                </div>
                """, unsafe_allow_html=True)

            else:

                violations_text = ", ".join(
                    sorted(set(r["violations"]))
                )

                st.markdown(
                    f"""
                    <div class="violation-card">
                        🔴 PPE VIOLATIONS DETECTED
                        <br>
                        <span style="font-weight:500;">
                        {violations_text}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # Metrics
            m1, m2 = st.columns(2)

            with m1:
                st.metric(
                    "Objects Detected",
                    len(last["dets"])
                )

            with m2:
                st.metric(
                    "Violations",
                    len(r["violations"])
                )


            st.markdown(
                '<div class="section-label">Detected Objects</div>',
                unsafe_allow_html=True
            )

            st.write(r["counts"])


        st.markdown("""
        <div class="info-card">
            🤖 <b>AI Assistant Ready</b><br>
            This inspection result is available in
            <b>Ask AI</b> for further safety analysis.
        </div>
        """, unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════
# 📄 DOCUMENTS  (Milestone 2)
# ══════════════════════════════════════════════════════════
elif page == "📄 Documents":
    st.header("📄 Document Knowledge Base")
    files = st.file_uploader("Upload manuals / inspection or incident reports",
                             type=["pdf", "txt", "docx"], accept_multiple_files=True)
    if files and st.button("Add to knowledge base", type="primary"):
        for f in files:
            chunks = process_file(save_upload(f))
            if chunks:
                kb.add_document(chunks, f.name)
                st.success(f"{f.name}: {len(chunks)} chunks added")
            else:
                st.warning(f"{f.name}: no text found (scanned PDF?)")

    st.subheader("Documents in the knowledge base")
    docs = kb.list_documents()
    if docs:
        st.table([{"document": name, "chunks": n} for name, n in docs.items()])
    else:
        st.info("No documents yet.")

    st.subheader("🔍 Test search")
    q = st.text_input("Search", placeholder="When must goggles be worn?")
    if q:
        for r in kb.search(q):
            st.info(f"**{r['source']} #{r['chunk']}** (distance {r['distance']})\n\n{r['text']}")


# ══════════════════════════════════════════════════════════
# 🤖 ASK AI  (Milestone 3 - LangGraph agent)
# ══════════════════════════════════════════════════════════
elif page == "🤖 Ask AI":

    st.markdown("""
    <div class="vd-hero">
        <h1>🤖 Safety AI Assistant</h1>
        <p>
            Ask questions about workplace safety,
            PPE requirements and inspection findings.
        </p>
    </div>
    """, unsafe_allow_html=True)
    last = st.session_state.get("last")
    use_image = False
    if last:
        use_image = st.checkbox(f"Include last detection result ({last['name']})", value=True)
        if use_image:
            st.caption(last["result"]["text"])

    question = st.text_input(
    "Your question",
    placeholder="Is this worker compliant? Which rule applies?"
)

    if st.button("Ask", type="primary"):

        if not question.strip():
            st.warning("⚠️ Please enter a question before clicking Ask.")

        else:
            vis = last["result"] if (use_image and last) else None

            with st.spinner("Thinking..."):
                out = ask_agent(
                question,
                vision=vis["text"] if vis else "",
                violations=vis["violations"] if vis else []
            )

            st.session_state["answer"] = {
            "question": question,
            **out
        }

            st.session_state["voted"] = False

    ans = st.session_state.get("answer")
    if ans:
        st.markdown("### Answer")
        st.write(ans["answer"])
        with st.expander("How the agent worked"):
            st.write("**Steps:** " + " → ".join(ans["steps"]))
            for d in ans["docs"]:
                st.caption(f"[{d['source']} #{d['chunk']}] {d['text'][:200]}")

        if not st.session_state.get("voted"):
            st.write("Was this answer helpful?")
            c1, c2 = st.columns(2)
            if c1.button("👍 Yes"):
                storage.log_feedback(ans["question"], True)
                st.session_state["voted"] = True
                st.rerun()
            if c2.button("👎 No"):
                storage.log_feedback(ans["question"], False)
                st.session_state["voted"] = True
                st.rerun()
        else:
            st.caption("Thanks for your feedback!")

# ══════════════════════════════════════════════════════════
# 📊 DASHBOARD  (Milestone 4)
# ══════════════════════════════════════════════════════════
elif page == "📊 Dashboard":

    # ---------- Dashboard Header ----------
    st.markdown("""
    <div class="vd-hero">
        <h1>📊 Safety Analytics Dashboard</h1>
        <p>
            Monitor workplace safety performance, compliance,
            violations and inspection activity.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Load existing data ----------
    scans = storage.load_scans()
    k = storage.kpis(scans)

    # ---------- KPI CARDS ----------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "📷 Total Scans",
            k["scans"]
        )

    with c2:
        st.metric(
            "🛡️ Compliance",
            f"{k['compliance']}%"
            if k["compliance"] is not None else "-"
        )

    with c3:
        st.metric(
            "🚨 Violations",
            k["violations"]
        )

    with c4:
        st.metric(
            "⭐ Satisfaction",
            f"{k['satisfaction']}%"
            if k["satisfaction"] is not None else "-",
            help="Target: 85% or more"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- NO DATA ----------
    if scans.empty:

        st.markdown("""
        <div class="vd-card" style="text-align:center;">

            <div style="font-size:3rem;">📊</div>

            <h2>No inspection data yet</h2>

            <p>
                Analyse an image or video from the
                PPE Detection page to start building
                your safety analytics.
            </p>

        </div>
        """, unsafe_allow_html=True)

    # ---------- DATA AVAILABLE ----------
    else:

        st.markdown(
            '<div class="section-title">Safety Analytics</div>',
            unsafe_allow_html=True
        )

        left, right = st.columns(2)

        # ---------- Violations by Type ----------
        with left:

            st.subheader("🚨 Violations by Type")
            st.caption("Breakdown of detected PPE violations.")

            counts = storage.violation_counts(scans)

            if len(counts):
                st.bar_chart(counts)

            else:
                st.success(
                    "✅ No violations detected so far."
                )

        # ---------- Violations Per Day ----------
        with right:

            st.subheader("📈 Violations Over Time")
            st.caption("Daily safety violation trend.")

            daily = (
                scans
                .set_index("time")["violations"]
                .resample("D")
                .sum()
            )

            if daily.sum() > 0:
                st.line_chart(daily)

            else:
                st.info(
    "🎉 No violations recorded. "
    "Your current inspections show a compliant safety status."
        )

        # ---------- Scan History ----------
        st.markdown(
            '<div class="section-title">Recent Inspections</div>',
            unsafe_allow_html=True
        )

        st.subheader("📋 Inspection History")
        st.caption(
        "Review previously analysed workplace images and videos."
        )

        st.dataframe(
            scans.sort_values(
                "time",
                ascending=False
            ),
            hide_index=True,
            use_container_width=True
        )

    # ---------- Clear History ----------
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Inspection History"):

        storage.clear_all()

        st.success(
            "Inspection history cleared successfully."
        )

        st.rerun()
# ══════════════════════════════════════════════════════════
# 📋 REPORT  (Milestone 4)
# ══════════════════════════════════════════════════════════
elif page == "📋 Report":

    st.markdown("""
    <div class="vd-card">
        <h2>📋 Compliance Report</h2>
        <p>
            Generate a professional workplace safety
            compliance report from the latest inspection data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    scans = storage.load_scans()

    if st.button("📄 Generate Compliance Report", type="primary"):
        k = storage.kpis(scans)
        counts = storage.violation_counts(scans)
        with st.spinner("Writing summary..."):
            summary = write_summary(k, counts)
        st.session_state["report"] = {"summary": summary, "pdf": build_pdf(k, counts, scans, summary)}

    rep = st.session_state.get("report")
    if rep:
        st.subheader("Summary")
        st.write(rep["summary"])
        st.download_button("⬇️ Download PDF", rep["pdf"],
                           file_name="safety_compliance_report.pdf", mime="application/pdf")
        st.download_button("⬇️ Download data (CSV)", scans.to_csv(index=False),
                           file_name="scans.csv", mime="text/csv")
