# 🤖 AI Question Master

Transform your documents and text into interactive learning materials using state-of-the-art Transformer models. This project provides an end-to-end pipeline for generating high-quality MCQs, a dedicated timed quiz screen, study guides, and an interactive Q&A assistant.

## 🌐 Live Demo

Try the hosted app here: https://ai-question-master.streamlit.app/

## 🚀 Key Features

- **📄 Multi-Source Input**: Support for raw text pasting, PDF (including multi-column layouts), and TXT files.
- **🎮 Dedicated Quiz Screen**: Opens quiz mode in a separate UI with question count, timer setup, auto-submit, and score review.
- **⏱️ Timed Quiz**: Countdown timer decreases live and automatically submits when it reaches zero.
- **✅ Answer Review**: Shows the correct option, the user’s selected option, and the final score after submission.
- **📖 Study Guide**: Unified view of key facts extracted from text and exploratory questions with revealable answers.
- **💡 Smart Assistant**: A chat-style interface for asking custom questions about your documents with hallucination detection.
- **⚡ Unified UI**: Single-click generation for all models (Answer-Aware, End-to-End, and QA).
- **🛠️ Robust Extraction**: Uses `pdfplumber` for layout-aware text extraction from complex PDFs.

## 🛠️ Setup & Installation

### 1. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```
The app will be available at `http://localhost:8501`.

Live hosted app: https://ai-question-master.streamlit.app/

If you are using Windows PowerShell and the project virtual environment, activate it first:
```powershell
.\.venv\Scripts\Activate.ps1
```

## 🧠 Project Architecture

This project utilizes fine-tuned **T5 (Text-to-Text Transfer Transformer)** models for multiple NLP tasks:

1.  **Answer Extraction**: Identifies candidate answer spans within the context.
2.  **Question Generation (QG)**: Generates questions specifically tailored to the extracted answers using the "Highlight" format.
3.  **End-to-End QG**: Generates broader, exploratory questions directly from the context.
4.  **Question Answering (QA)**: Provides answers to both generated and user-defined questions.
5.  **Timed Quiz Flow**: Moves selected generated questions into a dedicated quiz screen with live countdown and auto-submit.

### MCQ Generation Logic
Distractors are generated using a multi-step heuristic:
-   **WordNet Integration**: Finds semantically related "cousin" words (hyponyms of shared hypernyms).
-   **Contextual Fallback**: Uses other potential answers found in the document.
-   **Smart Fillers**: Logical placeholders for edge cases.

## 📂 Project Structure

- `app.py`: The primary Streamlit UI and application logic.
- `pipelines.py`: Core inference logic for all QG and QA tasks.
- `utils.py`: Utility functions for data processing and distractor generation.
- `notebooks/`: Exploration and training notebooks.
- `requirements.txt`: List of required Python packages.

## 🧪 Quiz Flow

1. Paste text or upload a document.
2. Click `Generate Everything` to create questions.
3. Click `Take Quiz` to open the dedicated quiz page.
4. Set the number of questions and the timer.
5. Start the quiz and answer the questions before the timer reaches zero.
6. Submit manually or let the app auto-submit when time runs out.
7. Review your score, selected answers, and correct answers.

## 🤝 Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📝 Usage Tips

-   **Context Length**: For best results, use paragraphs of 3-5 sentences.
-   **PDF Quality**: Ensure PDFs are text-based (not scanned images) for accurate extraction.
-   **Settings**: Use the "Max Questions" slider to control the volume of generated content.
-   **Quiz Mode**: Use `Take Quiz` to switch into the dedicated timed quiz screen.
-   **Timer**: The quiz auto-submits when time reaches zero, so keep an eye on the countdown.

## 📜 Acknowledgments

This project is built using the 🤗 [Transformers](https://github.com/huggingface/transformers) library and Streamlit.
