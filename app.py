import streamlit as st
from pipelines import pipeline
import nltk
import os
import random
import re
import time

# Ensure nltk resources are available
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

def download_nltk_resources():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    for find_path, download_name in resources:
        try:
            nltk.data.find(find_path, paths=[nltk_data_path])
        except LookupError:
            with st.spinner(f"Downloading NLTK resource: {download_name}..."):
                nltk.download(download_name, download_dir=nltk_data_path)

download_nltk_resources()

# Verify nltk works
def verify_nltk(text):
    try:
        sents = nltk.sent_tokenize(text)
        return sents
    except Exception as e:
        st.error(f"NLTK Error: {e}")
        return []

def normalize_question(q):
    # Lowercase, remove non-alphanumeric, and strip
    return re.sub(r'[^a-z0-9]', '', q.lower().strip())

def clear_quiz_state():
    for key in [key for key in st.session_state.keys() if key.startswith("quiz_choice_")]:
        del st.session_state[key]

def build_quiz_items(context, question_count):
    if st.session_state.qg_results and len(st.session_state.qg_results) >= question_count:
        source_questions = st.session_state.qg_results[:question_count]
    else:
        qg_nlp = load_pipeline("question-generation")
        source_questions = qg_nlp(context)[:question_count]
        st.session_state.qg_results = source_questions

    quiz_items = []
    for index, item in enumerate(source_questions):
        options = [item['answer']] + item['distractors']
        random.Random(42 + index).shuffle(options)
        quiz_items.append({
            'question': item['question'],
            'answer': item['answer'],
            'options': options,
        })
    return quiz_items

def finalize_quiz():
    review = []
    score = 0

    for index, item in enumerate(st.session_state.quiz_items):
        selected = st.session_state.get(f"quiz_choice_{index}")
        is_correct = selected == item['answer']
        if is_correct:
            score += 1

        review.append({
            'question': item['question'],
            'correct_answer': item['answer'],
            'selected_answer': selected,
            'is_correct': is_correct,
        })

    st.session_state.quiz_review = review
    st.session_state.quiz_score = score
    st.session_state.quiz_submitted = True
    st.session_state.quiz_active = False
    st.session_state.quiz_end_time = 0

def start_quiz(context, question_count, duration_minutes):
    if not context:
        st.warning("Please add a context before starting the quiz.")
        return

    sents = verify_nltk(context)
    if not sents:
        st.error("Failed to tokenize context into sentences.")
        return

    with st.spinner("Preparing your timed quiz..."):
        try:
            clear_quiz_state()
            st.session_state.quiz_items = build_quiz_items(context, question_count)
            if not st.session_state.quiz_items:
                st.info("No quiz questions could be generated from this context.")
                return

            st.session_state.quiz_total = len(st.session_state.quiz_items)
            st.session_state.quiz_score = 0
            st.session_state.quiz_review = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_active = True
            st.session_state.quiz_started_at = time.time()
            st.session_state.quiz_duration_seconds = int(duration_minutes * 60)
            st.session_state.quiz_end_time = st.session_state.quiz_started_at + st.session_state.quiz_duration_seconds
            st.session_state.app_view = "quiz"
            st.rerun()
        except Exception as exc:
            st.error(f"Quiz setup failed: {exc}")

@st.fragment(run_every=1)
def render_quiz_panel():
    if not st.session_state.get('quiz_items'):
        return

    total_seconds = int(st.session_state.get('quiz_duration_seconds', 0))
    remaining_seconds = max(0, int(st.session_state.get('quiz_end_time', 0) - time.time()))
    elapsed_seconds = max(0, total_seconds - remaining_seconds) if total_seconds else 0

    st.markdown("## 🎯 Timed Quiz")
    status_col, timer_col, score_col = st.columns(3)
    with status_col:
        st.metric("Questions", st.session_state.quiz_total)
    with timer_col:
        st.metric("Time Left", f"{remaining_seconds // 60:02d}:{remaining_seconds % 60:02d}")
    with score_col:
        if st.session_state.quiz_submitted:
            st.metric("Score", f"{st.session_state.quiz_score}/{st.session_state.quiz_total}")
        else:
            st.metric("Progress", f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d}")

    if st.session_state.quiz_submitted:
        st.success(f"Quiz completed. Final score: {st.session_state.quiz_score}/{st.session_state.quiz_total}")
    elif remaining_seconds <= 0:
        st.warning("Time is up. Submitting your quiz now.")
        finalize_quiz()
        st.rerun()

    if not st.session_state.quiz_submitted:
        progress = 0.0 if not total_seconds else min(1.0, elapsed_seconds / total_seconds)
        st.progress(progress)

    for index, item in enumerate(st.session_state.quiz_items):
        with st.container(border=True):
            st.markdown(f"### Question {index + 1}")
            st.write(item['question'])
            st.radio(
                "Choose your answer",
                item['options'],
                key=f"quiz_choice_{index}",
                index=None,
                disabled=st.session_state.quiz_submitted,
            )

    action_col, reset_col = st.columns([1, 1])
    with action_col:
        if not st.session_state.quiz_submitted and st.button("Submit Quiz", type="primary"):
            finalize_quiz()
            st.rerun()
    with reset_col:
        if st.button("Start New Quiz"):
            clear_quiz_state()
            st.session_state.quiz_items = []
            st.session_state.quiz_review = []
            st.session_state.quiz_total = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_submitted = False
            st.session_state.quiz_active = False
            st.session_state.quiz_duration_seconds = 0
            st.session_state.quiz_end_time = 0
            st.rerun()

    if st.session_state.quiz_submitted:
        st.markdown("### Review")
        for index, result in enumerate(st.session_state.quiz_review):
            with st.container(border=True):
                st.markdown(f"**Question {index + 1}:** {result['question']}")
                st.write(f"**Your choice:** {result['selected_answer'] or 'No answer selected'}")
                st.write(f"**Correct answer:** {result['correct_answer']}")
                if result['is_correct']:
                    st.success("Correct")
                else:
                    st.error("Incorrect")

st.set_page_config(
    page_title="AI Question Master",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #d62828 !important;
        color: white !important;
        border-color: #d62828 !important;
    }
    .stTextArea>div>div>textarea {
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 AI Question Master")
st.markdown("---")

# Initialize session state for MCQs
if 'qg_results' not in st.session_state:
    st.session_state.qg_results = []
if 'e2e_results' not in st.session_state:
    st.session_state.e2e_results = []
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'app_view' not in st.session_state:
    st.session_state.app_view = "home"
if 'quiz_context' not in st.session_state:
    st.session_state.quiz_context = ""
if 'quiz_items' not in st.session_state:
    st.session_state.quiz_items = []
if 'quiz_review' not in st.session_state:
    st.session_state.quiz_review = []
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'quiz_total' not in st.session_state:
    st.session_state.quiz_total = 0
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'quiz_active' not in st.session_state:
    st.session_state.quiz_active = False
if 'quiz_duration_seconds' not in st.session_state:
    st.session_state.quiz_duration_seconds = 0
if 'quiz_end_time' not in st.session_state:
    st.session_state.quiz_end_time = 0
if 'quiz_started_at' not in st.session_state:
    st.session_state.quiz_started_at = 0

@st.cache_resource
def load_pipeline(task):
    return pipeline(task)

# Sidebar with app info
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/brain.png")
    st.header("About")
    st.info("Transform your text into interactive learning materials using state-of-the-art AI models.")
    if st.button("Clear Session"):
        st.session_state.qg_results = []
        st.session_state.e2e_results = []
        st.session_state.qa_history = []
        st.session_state.quiz_context = ""
        st.session_state.app_view = "home"
        st.session_state.quiz_items = []
        st.session_state.quiz_review = []
        st.session_state.quiz_score = 0
        st.session_state.quiz_total = 0
        st.session_state.quiz_submitted = False
        st.session_state.quiz_active = False
        st.session_state.quiz_duration_seconds = 0
        st.session_state.quiz_end_time = 0
        st.session_state.quiz_started_at = 0
        st.rerun()

if st.session_state.app_view == "quiz":
    st.markdown("## 🎯 Quiz Arena")
    quiz_header_col, quiz_exit_col = st.columns([3, 1])
    with quiz_header_col:
        st.caption("This is a dedicated quiz screen with its own timer and review flow.")
    with quiz_exit_col:
        if st.button("← Back to Dashboard"):
            st.session_state.app_view = "home"
            st.rerun()

    if not st.session_state.quiz_items and not st.session_state.quiz_submitted:
        st.info("Set up your timed quiz below, then start it from this screen.")
        quiz_setup_col1, quiz_setup_col2 = st.columns(2)
        with quiz_setup_col1:
            quiz_question_count = st.number_input("Number of quiz questions", min_value=1, max_value=20, value=5, step=1)
        with quiz_setup_col2:
            quiz_duration_minutes = st.number_input("Timer (minutes)", min_value=1, max_value=60, value=5, step=1)

        if st.button("Start Quiz", type="primary"):
            start_quiz(st.session_state.quiz_context, int(quiz_question_count), int(quiz_duration_minutes))
    else:
        render_quiz_panel()

    st.stop()

# Main Layout
col_input, col_action = st.columns([3, 1])

with col_input:
    st.subheader("📄 Input Source")
    input_type = st.radio("Select Input Type", ["Text", "File (PDF/TXT)"], horizontal=True)
    
    context = ""
    if input_type == "Text":
        context = st.text_area("✍️ Paste Text", height=250, placeholder="Paste your paragraph here...", value="Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace.")
    else:
        uploaded_file = st.file_uploader("📂 Upload PDF or TXT", type=['pdf', 'txt'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.pdf'):
                    import pdfplumber
                    with pdfplumber.open(uploaded_file) as pdf:
                        file_text = ""
                        for page in pdf.pages:
                            text = page.extract_text(layout=True)
                            if text:
                                file_text += text + "\n"
                else:
                    file_text = uploaded_file.read().decode("utf-8")
                
                context = st.text_area("📝 Extracted Text (Editable)", value=file_text, height=250)
            except Exception as e:
                st.error(f"Error reading file: {e}")

with col_action:
    st.write("### ⚙️ Settings")
    max_questions = st.slider("Max Questions per Type", min_value=1, max_value=20, value=5)
    generate_btn = st.button("Generate Everything", type="primary")
    st.write("")
    st.write("")
    if st.button("Take Quiz"):
        if context:
            st.session_state.quiz_context = context
            st.session_state.app_view = "quiz"
            st.rerun()
        else:
            st.warning("Add some context first, then open the quiz page.")

if generate_btn:
    if context:
        sents = verify_nltk(context)
        if not sents:
            st.error("Failed to tokenize context into sentences.")
        else:
            with st.spinner("Hold on tight .. I am thinking..."):
                try:
                    qg_nlp = load_pipeline("question-generation")
                    e2e_nlp = load_pipeline("e2e-qg")
                    
                    st.session_state.qg_results = qg_nlp(context)[:max_questions]
                    e2e_raw = e2e_nlp(context)
                    
                    qg_questions_norm = {normalize_question(res['question']) for res in st.session_state.qg_results}
                    deduped_e2e = []
                    seen_e2e_norm = set()
                    for q in e2e_raw:
                        norm_q = normalize_question(q)
                        if norm_q not in qg_questions_norm and norm_q not in seen_e2e_norm:
                            deduped_e2e.append(q)
                            seen_e2e_norm.add(norm_q)
                    st.session_state.e2e_results = deduped_e2e[:max_questions]
                    
                    if not st.session_state.qg_results and not st.session_state.e2e_results:
                        st.info("No questions could be generated. Try a longer context.")
                except Exception as e:
                    st.error(f"Error: {e}")

# Result Sections in Tabs
if st.session_state.qg_results or st.session_state.e2e_results:
    st.write("## 📝 Results")
    tab1, tab2 = st.tabs(["📖 Study Guide", "💡 Custom QA"])

    with tab1:
        col_rec, col_exp = st.columns(2)
        with col_rec:
            st.subheader("🎯 Key Facts")
            for i, res in enumerate(st.session_state.qg_results):
                st.info(f"**Q:** {res['question']}\n\n**A:** {res['answer']}")
        
        with col_exp:
            st.subheader("🔍 Explore More")
            qa_nlp = load_pipeline("multitask-qa-qg")
            for i, q in enumerate(st.session_state.e2e_results):
                with st.expander(f"Question: {q}"):
                    if st.button("Reveal Answer", key=f"reveal_{i}"):
                        ans = qa_nlp({"question": q, "context": context})
                        st.write(f"**Answer:** {ans}")

    with tab2:
        st.subheader("� Smart Assistant")
        user_q = st.text_input("Ask a specific question about your text:", placeholder="e.g., Who is Guido?")
        if st.button("Get Instant Answer"):
            if user_q:
                qa_nlp = load_pipeline("multitask-qa-qg")
                with st.spinner("Finding answer..."):
                    ans = qa_nlp({"question": user_q, "context": context})
                    st.session_state.qa_history.append({"q": user_q, "a": ans})
            else:
                st.warning("Please type a question.")
        
        for item in reversed(st.session_state.qa_history):
            with st.chat_message("user"):
                st.write(item['q'])
            with st.chat_message("assistant"):
                st.write(item['a'])
