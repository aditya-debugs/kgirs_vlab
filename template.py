import json
import os
import re
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF


# =============================================================================
# 1. EXPERIMENT CONFIGURATION
# =============================================================================

EXPERIMENT_CONFIG = {
    "title": "Document Processing Pipeline for Information Retrieval",
    "subtitle": "Virtual Laboratory Experiment",
    "objectives": [
        "Understand the end-to-end lifecycle of textual documents in an Information Retrieval (IR) system.",
        "Acquire pre-extracted textual documents from a simulated document collection.",
        "Apply text cleaning and normalization operations to raw document text.",
        "Tokenize documents and remove common stop words.",
        "Apply stemming to reduce related word forms to common stems.",
        "Prepare a processed document representation suitable for retrieval and indexing."
    ]
}

THEORY_CONTENT = {
    "background": """
### Overview

Information Retrieval (IR) is concerned with storing, organizing, processing and retrieving
information from a collection of documents. Before a document can be efficiently searched,
its raw textual content is normally transformed into a standardized representation.

In this experiment, the document collection is simulated using a JSON file. The JSON file
contains PDF filenames and their already-extracted textual content. The PDF files themselves
are not required for execution; the extracted text represents the acquisition stage.

### Document Processing Pipeline

The experiment follows the sequence:

**Document Collection → Acquisition → Cleaning → Tokenization → Stop-word Removal → Stemming → Retrieval-ready Terms**

Each stage transforms the document while preserving the terms that are useful for subsequent
retrieval or indexing.

### Why preprocessing is required

Raw documents may contain punctuation, inconsistent capitalization, numbers, extra whitespace
and very common words. Preprocessing reduces such variation and produces a cleaner term
representation for an IR system.

### Workflow

1. Select a document from the simulated collection.
2. Acquire its pre-extracted text.
3. Clean and normalize the text.
4. Tokenize the normalized text.
5. Remove stop words.
6. Apply stemming.
7. Inspect the final retrieval-ready representation.
8. Record the trial and compare different preprocessing configurations.
    """,

    "key_terms": {
        "Document Collection": "A set of documents available to the IR system.",
        "Acquisition": "Obtaining the textual representation of a document for processing.",
        "Cleaning": "Removing unwanted characters, punctuation, extra spaces and other noise.",
        "Tokenization": "Dividing text into individual tokens, generally words or terms.",
        "Stop Words": "Very common words that may contribute relatively little to retrieval in a basic term-based pipeline.",
        "Stemming": "Reducing related word forms to a common stem using heuristic rules.",
        "Retrieval-ready Representation": "The processed set of terms that can subsequently be used for indexing and retrieval."
    },

    "procedure": [
        "Open the Simulation section.",
        "Select a document from the simulated document collection.",
        "Choose the preprocessing operations to apply.",
        "Run the document processing pipeline.",
        "Observe the output after each processing stage.",
        "Compare the number of terms before and after preprocessing.",
        "Record at least three trials using different preprocessing configurations.",
        "Complete the conceptual quiz.",
        "Generate the final experiment report."
    ],

    "references": [
        "C. D. Manning, P. Raghavan, and H. Schutze, Introduction to Information Retrieval, Cambridge University Press, 2008.",
        "R. Baeza-Yates and B. Ribeiro-Neto, Modern Information Retrieval, 2nd Edition, Addison-Wesley, 2011.",
        "M. F. Porter, \"An algorithm for suffix stripping,\" Program, vol. 14, no. 3, pp. 130-137, 1980.",
        "IIT Kharagpur Virtual Labs, Information Retrieval Lab, https://vlab.co.in/"
    ]
}


# =============================================================================
# 2. SIMULATED DOCUMENT COLLECTION
# =============================================================================

DATA_FILE = "document_collection.json"


def load_document_collection():
    if not os.path.exists(DATA_FILE):
        st.error(
            f"Missing {DATA_FILE}. Place it in the same directory as template.py."
        )
        st.stop()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


DOCUMENT_COLLECTION = load_document_collection()["documents"]


# =============================================================================
# 3. TEXT PROCESSING ENGINE
# =============================================================================

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i",
    "if", "in", "into", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "she", "that", "the", "their", "there", "these",
    "they", "this", "to", "was", "we", "were", "which", "with",
    "you", "your", "will", "would", "can", "could", "should",
    "than", "then", "also", "using", "used"
}


def clean_text(text):
    """Lowercase, remove URLs, punctuation/numbers and normalize whitespace."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """Simple word/token tokenizer."""
    return re.findall(r"\b[a-z]+\b", text.lower())


def remove_stop_words(tokens):
    return [token for token in tokens if token not in STOP_WORDS]


def stem_word(word):
    """
    Small educational rule-based stemmer.
    This is intentionally transparent so students can observe the transformation.
    """
    if len(word) <= 3:
        return word

    suffixes = [
        "ingly", "edly", "ation", "ness",
        "ing", "ers", "ies", "ied", "ed", "es", "ly", "s"
    ]

    for suffix in suffixes:
        if suffix == "s" and word.endswith(("ss", "us", "is")):
            continue

        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            root = word[:-len(suffix)]
            if suffix in ("ies", "ied"):
                return root + "y"
            return root

    return word


def stem_tokens(tokens):
    return [stem_word(token) for token in tokens]


def process_document(
    raw_text,
    do_clean=True,
    do_stopword=True,
    do_stemming=True
):
    """Runs the complete configurable document-processing pipeline."""

    acquired = raw_text

    cleaned = clean_text(acquired) if do_clean else acquired
    tokenized = tokenize(cleaned)

    filtered = (
        remove_stop_words(tokenized)
        if do_stopword
        else tokenized
    )

    stemmed = stem_tokens(filtered) if do_stemming else filtered

    final_terms = stemmed

    return {
        "acquired": acquired,
        "cleaned": cleaned,
        "tokens": tokenized,
        "filtered_tokens": filtered,
        "stemmed_tokens": stemmed,
        "final_terms": final_terms
    }


# =============================================================================
# 4. SIMULATION METRICS
# =============================================================================

def calculate_metrics(result):
    original_tokens = tokenize(result["acquired"])
    final_terms = result["final_terms"]

    return {
        "Original Words": len(original_tokens),
        "Final Terms": len(final_terms),
        "Unique Terms": len(set(final_terms)),
        "Terms Removed": max(0, len(original_tokens) - len(final_terms)),
        "Reduction (%)": round(
            ((len(original_tokens) - len(final_terms)) /
             len(original_tokens) * 100),
            2
        ) if original_tokens else 0.0
    }


# =============================================================================
# 5. PDF REPORT GENERATOR
# =============================================================================

class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(
            0, 10,
            f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Report",
            align="C"
        )


def safe_pdf_text(value):
    """Keep report text compatible with the built-in Helvetica font."""
    return str(value).encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(
    student_name,
    student_id,
    date_str,
    selected_document,
    trials_df,
    quiz_score,
    quiz_total,
    student_notes
):
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(30, 58, 138)
    pdf.multi_cell(
        0, 8,
        safe_pdf_text(EXPERIMENT_CONFIG["title"]),
        align="L"
    )
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(
        0, 5,
        safe_pdf_text(
            f"Student: {student_name or 'N/A'}    |    "
            f"Roll / ID: {student_id or 'N/A'}    |    "
            f"Date: {date_str}"
        )
    )
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Learning Objectives", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        pdf.multi_cell(
            0, 5,
            safe_pdf_text("- " + obj),
            new_x="LMARGIN",
            new_y="NEXT"
        )
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(
        0, 7,
        "2. Selected Document",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 5, safe_pdf_text(selected_document))
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(
        0, 7,
        "3. Recorded Experimental Trials",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(90, 90, 90)
        pdf.cell(
            0, 6,
            "No trials recorded during this session.",
            new_x="LMARGIN",
            new_y="NEXT"
        )
    else:
        cols = list(trials_df.columns)
        col_w = max(25, int(190 / len(cols)))

        pdf.set_font("Helvetica", "B", 7)
        pdf.set_fill_color(220, 228, 240)
        pdf.set_text_color(20, 20, 20)

        for col in cols:
            pdf.cell(
                col_w, 6, safe_pdf_text(str(col)[:16]),
                border=1, new_x="RIGHT", new_y="TOP",
                align="C", fill=True
            )
        pdf.ln()

        pdf.set_font("Helvetica", "", 7)

        for _, row in trials_df.iterrows():
            for col in cols:
                pdf.cell(
                    col_w, 5,
                    safe_pdf_text(str(row[col])[:16]),
                    border=1, new_x="RIGHT", new_y="TOP",
                    align="C"
                )
            pdf.ln()

    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(
        0, 7,
        "4. Observations & Analysis",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    notes = student_notes.strip() or (
        "The document-processing pipeline transformed the selected document "
        "through cleaning, tokenization, stop-word removal and stemming. "
        "The resulting terms provide a normalized representation suitable "
        "for subsequent retrieval or indexing."
    )
    pdf.multi_cell(0, 5, safe_pdf_text(notes))
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(
        0, 7,
        "5. Quiz Evaluation",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    percentage = int((quiz_score / quiz_total) * 100) if quiz_total else 0
    pdf.cell(
        0, 6,
        safe_pdf_text(f"Score: {quiz_score} / {quiz_total} ({percentage}%)"),
        new_x="LMARGIN",
        new_y="NEXT"
    )

    return bytes(pdf.output())


# =============================================================================
# 6. QUIZ
# =============================================================================

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "What is the main purpose of document preprocessing in Information Retrieval?",
        "options": [
            "To prepare textual data into a suitable representation for retrieval",
            "To permanently delete the document",
            "To convert every document into an image",
            "To encrypt the document"
        ],
        "answer_index": 0,
        "explanation": "Preprocessing converts raw text into a cleaner and more consistent representation for subsequent retrieval or indexing."
    },
    {
        "id": 2,
        "question": "What is tokenization?",
        "options": [
            "Dividing text into individual tokens or terms",
            "Removing the complete document",
            "Encrypting individual words",
            "Sorting PDF files by size"
        ],
        "answer_index": 0,
        "explanation": "Tokenization divides textual content into smaller units such as words or terms."
    },
    {
        "id": 3,
        "question": "Why can stop-word removal be useful in a basic term-based IR pipeline?",
        "options": [
            "It removes common words that may contribute relatively little to retrieval",
            "It translates all words into another language",
            "It guarantees perfect retrieval",
            "It converts text into numerical images"
        ],
        "answer_index": 0,
        "explanation": "Very common words can add little discriminative value in some retrieval settings, so they may be removed."
    },
    {
        "id": 4,
        "question": "What does stemming attempt to do?",
        "options": [
            "Reduce related word forms to a common stem",
            "Create a PDF from every word",
            "Remove every noun from a document",
            "Rank documents by file size"
        ],
        "answer_index": 0,
        "explanation": "Stemming applies rules to reduce related word forms to a common stem."
    },
    {
        "id": 5,
        "question": "Which stage normally occurs before stop-word removal?",
        "options": [
            "Tokenization",
            "Report generation",
            "Quiz evaluation",
            "PDF downloading"
        ],
        "answer_index": 0,
        "explanation": "The text must first be divided into tokens so that individual words can be evaluated against the stop-word list."
    },
    {
        "id": 6,
        "question": "What is the role of text cleaning?",
        "options": [
            "Remove unwanted characters and normalize the textual representation",
            "Increase punctuation in the document",
            "Convert all words into stop words",
            "Generate the final quiz"
        ],
        "answer_index": 0,
        "explanation": "Cleaning reduces unwanted textual noise such as punctuation, URLs, numbers and inconsistent whitespace."
    },
    {
        "id": 7,
        "question": "What does a retrieval-ready representation contain?",
        "options": [
            "Processed terms that can subsequently be used for indexing and retrieval",
            "Only the original PDF filename",
            "Only punctuation marks",
            "Only stop words"
        ],
        "answer_index": 0,
        "explanation": "The processed representation provides terms that can be passed to later indexing and retrieval stages."
    },
    {
        "id": 8,
        "question": "Why should multiple preprocessing trials be recorded?",
        "options": [
            "To compare the effect of different preprocessing configurations",
            "To make the application slower",
            "To delete previous results",
            "To avoid observing intermediate stages"
        ],
        "answer_index": 0,
        "explanation": "Multiple trials allow the effect of cleaning, stop-word removal and stemming choices to be compared."
    },
    {
        "id": 9,
        "question": "Which of the following is a document collection in this experiment?",
        "options": [
            "The simulated set of PDF filenames and their extracted text stored in JSON",
            "The quiz answer sheet",
            "The generated PDF report",
            "The Streamlit sidebar"
        ],
        "answer_index": 0,
        "explanation": "The JSON file acts as the simulated document collection and stores the extracted text associated with each PDF filename."
    },
    {
        "id": 10,
        "question": "What is the final purpose of this experiment's pipeline?",
        "options": [
            "Prepare documents for subsequent retrieval or indexing",
            "Replace the entire IR system",
            "Generate only PDF reports",
            "Remove all textual information"
        ],
        "answer_index": 0,
        "explanation": "The experiment focuses on document preparation; retrieval and indexing can use the resulting processed representation."
    }
]


# =============================================================================
# 7. STREAMLIT SECTIONS
# =============================================================================

def render_theory_section():
    st.header("Theoretical Framework & Background")
    st.markdown(THEORY_CONTENT["background"])

    st.divider()

    st.subheader("Learning Objectives")
    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"], 1):
        st.write(f"**Objective {i}:** {obj}")

    st.divider()

    st.subheader("Experimental Procedure")
    for i, step in enumerate(THEORY_CONTENT["procedure"], 1):
        st.write(f"**Step {i}:** {step}")

    st.divider()

    with st.expander("Key Terminology & Variable Reference"):
        terms_df = pd.DataFrame(
            list(THEORY_CONTENT["key_terms"].items()),
            columns=["Term", "Definition"]
        )
        st.table(terms_df)

    st.divider()

    st.subheader("References")
    for ref in THEORY_CONTENT["references"]:
        st.write(f"- {ref}")

    st.info(
        "Note: The document collection is simulated. PDF filenames identify "
        "the documents, while their extracted textual content is stored in "
        "document_collection.json for the experiment."
    )


def render_pipeline_stage(title, content, height=180):
    st.markdown(f"### {title}")
    if isinstance(content, list):
        st.code(" ".join(content), language="text")
    else:
        st.text_area(
            title,
            value=str(content),
            height=height,
            disabled=True,
            label_visibility="collapsed"
        )


def render_simulation_section():
    st.header("Interactive Document Processing Pipeline")
    st.write(
        "Select a document from the simulated collection and configure the "
        "preprocessing stages. The application processes the pre-extracted "
        "text stored in the JSON collection."
    )

    pdf_names = [doc["pdf_name"] for doc in DOCUMENT_COLLECTION]
    selected_pdf = st.selectbox(
        "Select Document from Collection",
        pdf_names
    )

    selected_doc = next(
        doc for doc in DOCUMENT_COLLECTION
        if doc["pdf_name"] == selected_pdf
    )

    st.info(
        f"Selected document: **{selected_doc['pdf_name']}**"
    )

    with st.expander("View document description"):
        st.write(selected_doc["description"])

    st.subheader("Processing Configuration")

    c1, c2, c3 = st.columns(3)

    with c1:
        do_clean = st.checkbox("Text Cleaning & Normalization", value=True)

    with c2:
        do_stopword = st.checkbox("Stop-word Removal", value=True)

    with c3:
        do_stemming = st.checkbox("Stemming", value=True)

    st.caption(
        "The selected PDF name represents the document shown to the user. "
        "Its extracted text is loaded internally from the JSON collection."
    )

    result = process_document(
        selected_doc["text"],
        do_clean=do_clean,
        do_stopword=do_stopword,
        do_stemming=do_stemming
    )
    st.session_state["last_result"] = result
    st.session_state["last_pdf"] = selected_pdf
    st.session_state["last_config"] = {
        "Cleaning": do_clean,
        "Stop-word Removal": do_stopword,
        "Stemming": do_stemming
    }
    metrics = calculate_metrics(result)

    st.divider()
    st.subheader("Pipeline Execution")

    st.caption(
        "The output below updates automatically whenever you change the "
        "selected document or the processing configuration above."
    )

    st.markdown(
        f"**Document Acquisition → Cleaning → Tokenization → "
        f"Stop-word Removal → Stemming → Retrieval-ready Terms**"
    )

    with st.expander("1. Acquired Text — Extracted Document", expanded=True):
        st.text_area(
            "Acquired Text",
            selected_doc["text"],
            height=150,
            disabled=True,
            label_visibility="collapsed"
        )

    with st.expander("2. Cleaned & Normalized Text", expanded=True):
        st.code(result["cleaned"], language="text")

    with st.expander("3. Tokenization", expanded=True):
        st.write(f"**Token count:** {len(result['tokens'])}")
        st.code(" | ".join(result["tokens"]), language="text")

    with st.expander("4. Stop-word Removal", expanded=True):
        st.write(
            f"**Terms after stop-word removal:** "
            f"{len(result['filtered_tokens'])}"
        )
        st.code(
            " | ".join(result["filtered_tokens"]),
            language="text"
        )

    with st.expander("5. Stemming", expanded=True):
        st.code(
            " | ".join(result["stemmed_tokens"]),
            language="text"
        )

    with st.expander("6. Final Retrieval-ready Representation", expanded=True):
        st.success(
            "The processed terms can now be passed to a later indexing/retrieval stage."
        )
        st.code(
            " ".join(result["final_terms"]),
            language="text"
        )

    st.divider()
    st.subheader("Experimental Metrics")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Original Words", metrics["Original Words"])
    m2.metric("Final Terms", metrics["Final Terms"])
    m3.metric("Unique Terms", metrics["Unique Terms"])
    m4.metric("Reduction", f"{metrics['Reduction (%)']}%")

    # Simple visual comparison
    stage_names = [
        "Original",
        "Tokenized",
        "After Stop Words",
        "Final"
    ]
    stage_values = [
        metrics["Original Words"],
        len(result["tokens"]),
        len(result["filtered_tokens"]),
        len(result["final_terms"])
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=stage_names,
            y=stage_values,
            text=stage_values,
            textposition="auto"
        )
    )
    fig.update_layout(
        title="Term Count Across Processing Stages",
        xaxis_title="Processing Stage",
        yaxis_title="Number of Terms",
        height=400
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Term Frequency Preview")

    frequency = Counter(result["final_terms"])
    freq_df = pd.DataFrame(
        frequency.most_common(15),
        columns=["Term", "Frequency"]
    )

    if not freq_df.empty:
        st.dataframe(
            freq_df,
            hide_index=True,
            width="stretch"
        )

    st.divider()
    st.subheader("Experimental Data Log Book")

    if st.button(
        "Record Current Trial",
        type="primary",
        width="stretch"
    ):
        trial = {
            "Trial #": len(st.session_state["trials"]) + 1,
            "PDF": st.session_state.get("last_pdf", selected_pdf),
            "Cleaning": "Yes" if do_clean else "No",
            "Stop Words": "Yes" if do_stopword else "No",
            "Stemming": "Yes" if do_stemming else "No",
            "Original": metrics["Original Words"],
            "Final": metrics["Final Terms"],
            "Unique": metrics["Unique Terms"],
            "Reduction %": metrics["Reduction (%)"],
            "Time": datetime.now().strftime("%H:%M:%S")
        }

        st.session_state["trials"].append(trial)
        st.success(f"Trial #{trial['Trial #']} recorded.")

    if st.button("Clear Logged Trials"):
        st.session_state["trials"] = []

    if st.session_state["trials"]:
        trials_df = pd.DataFrame(st.session_state["trials"])
        st.dataframe(
            trials_df,
            hide_index=True,
            width="stretch"
        )

        csv_data = trials_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Trial Data as CSV",
            data=csv_data,
            file_name="document_processing_trials.csv",
            mime="text/csv",
            width="stretch"
        )
    else:
        st.info(
            "No trials recorded yet. Run the pipeline and record at least "
            "three different configurations for comparison."
        )


def render_quiz_section():
    st.header("Concept Assessment Quiz")
    st.write(
        "Answer the following questions to evaluate your understanding "
        "of the document-processing pipeline."
    )

    with st.form("ir_quiz_form"):
        responses = {}

        for q in QUIZ_QUESTIONS:
            st.subheader(f"Question {q['id']}")
            selected = st.radio(
                q["question"],
                q["options"],
                key=f"q_{q['id']}"
            )
            responses[q["id"]] = q["options"].index(selected)

        submitted = st.form_submit_button(
            "Submit Quiz for Grading",
            type="primary"
        )

    if submitted:
        score = 0

        for q in QUIZ_QUESTIONS:
            if responses[q["id"]] == q["answer_index"]:
                score += 1
                st.success(
                    f"Question {q['id']}: Correct — {q['explanation']}"
                )
            else:
                st.error(
                    f"Question {q['id']}: Incorrect. "
                    f"Correct answer: {q['options'][q['answer_index']]}. "
                    f"{q['explanation']}"
                )

        st.session_state["quiz_score"] = score
        st.session_state["quiz_submitted"] = True

        percentage = int((score / len(QUIZ_QUESTIONS)) * 100)
        st.info(
            f"Final Score: **{score}/{len(QUIZ_QUESTIONS)} ({percentage}%)**"
        )

    elif st.session_state.get("quiz_submitted", False):
        score = st.session_state.get("quiz_score", 0)
        st.success(
            f"Quiz submitted. Current score: "
            f"**{score}/{len(QUIZ_QUESTIONS)}**"
        )


def render_report_section():
    st.header("Report Generation")
    st.write(
        "Compile your student information, recorded trials, observations "
        "and quiz evaluation into a PDF laboratory report."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        student_name = st.text_input(
            "Student Name",
            value=st.session_state["student_info"].get(
                "name", "Student Name"
            )
        )

    with c2:
        student_id = st.text_input(
            "Student Roll / ID",
            value=st.session_state["student_info"].get(
                "id", "EXP-001"
            )
        )

    with c3:
        lab_date = st.date_input(
            "Experiment Date",
            value=datetime.now().date()
        )

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["date"] = str(lab_date)

    st.subheader("Discussion & Observations")

    notes = st.text_area(
        "Enter your observations and conclusion",
        value=st.session_state.get(
            "student_notes",
            "The document-processing pipeline transformed the selected "
            "document into a normalized retrieval-ready representation. "
            "Preprocessing reduced unnecessary textual variation and "
            "provided a set of terms suitable for subsequent indexing "
            "and retrieval."
        ),
        height=150
    )

    st.session_state["student_notes"] = notes

    trials_df = (
        pd.DataFrame(st.session_state["trials"])
        if st.session_state["trials"]
        else pd.DataFrame()
    )

    st.divider()
    st.subheader("Report Summary Preview")

    st.write(f"**Experiment:** {EXPERIMENT_CONFIG['title']}")
    st.write(f"**Student:** {student_name}")
    st.write(f"**Roll / ID:** {student_id}")
    st.write(f"**Date:** {lab_date}")

    selected = st.session_state.get("last_pdf", "No document processed yet")
    st.write(f"**Last Processed Document:** {selected}")

    score = st.session_state.get("quiz_score", 0)
    st.write(
        f"**Quiz Score:** {score}/{len(QUIZ_QUESTIONS)}"
    )

    if not trials_df.empty:
        st.dataframe(
            trials_df,
            hide_index=True,
            width="stretch"
        )
    else:
        st.info("No experimental trials have been recorded yet.")

    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        selected_document=selected,
        trials_df=trials_df,
        quiz_score=score,
        quiz_total=len(QUIZ_QUESTIONS),
        student_notes=notes
    )

    st.divider()

    st.download_button(
        "Download Official Lab Report (.pdf)",
        data=pdf_bytes,
        file_name="IR_Document_Processing_Lab_Report.pdf",
        mime="application/pdf",
        type="primary",
        width="stretch"
    )


# =============================================================================
# 8. SESSION STATE & MAIN ENTRYPOINT
# =============================================================================

def init_session_state():
    if "trials" not in st.session_state:
        st.session_state["trials"] = []

    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0

    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False

    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP-001",
            "date": str(datetime.now().date())
        }

    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""


def main():
    st.set_page_config(
        page_title=EXPERIMENT_CONFIG["title"],
        page_icon="📚",
        layout="wide"
    )

    init_session_state()

    st.title(EXPERIMENT_CONFIG["title"])
    st.caption(EXPERIMENT_CONFIG["subtitle"])

    section = st.sidebar.radio(
        "Lab Navigator",
        [
            "Theory",
            "Simulation",
            "Quiz",
            "Report Generation"
        ]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Experiment Progress")

    st.sidebar.write(
        f"Trials Recorded: **{len(st.session_state['trials'])}**"
    )

    quiz_status = (
        "Completed"
        if st.session_state.get("quiz_submitted", False)
        else "Pending"
    )
    st.sidebar.write(f"Quiz: **{quiz_status}**")

    if st.session_state.get("quiz_submitted", False):
        st.sidebar.write(
            f"Score: **{st.session_state['quiz_score']}/"
            f"{len(QUIZ_QUESTIONS)}**"
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Document collection: document_collection.json"
    )

    if section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()


if __name__ == "__main__":
    main()