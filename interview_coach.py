import os
import io
import json
import tempfile
import random
from dataclasses import dataclass, asdict
from typing import List, Dict
import streamlit as st

# Optional: use the modern OpenAI SDK (>=1.0.0)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------- App config ----------
st.set_page_config(page_title="AI Voice Interview Coach", page_icon="🎤", layout="wide")
st.title("🎤 AI Voice Interview Coach")
st.caption("Manual Mode: record → transcribe → submit → rating + feedback. Practice Set Mode: random questions with a final report.")

# Ensure OpenAI API key is set  (Note: for production, move this to secrets)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"  # Replace with your actual key

# ---------- Model configuration ----------
MODEL = "gpt-4o-mini"
TRANSCRIBE_MODEL = "whisper-1"

# ---------- Subjects & Questions ----------
# ---------- Load Questions from JSON ----------
import json
import os

QUESTIONS_FILE = "questions.json"

def load_question_bank(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        st.error(f"Questions file not found: {file_path}")
        return {}

QUESTION_BANK = load_question_bank(QUESTIONS_FILE)

# ---------- Data classes ----------
@dataclass
class QARecord:
    subject: str
    question: str
    transcript: str
    rating_10: int
    explanation: str
    feedback: str
    strengths: str
    improvements: str
    sample_answer: str

# ---------- Helpers ----------
def get_client():
    """Create an OpenAI client if possible."""
    if not OPENAI_AVAILABLE:
        return None
    key = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_BASE_URL")  # optional for compatible servers
    if not key and not base:
        return None
    try:
        return OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)
    except Exception:
        return None

def transcribe_audio_filebytes(file_bytes: bytes, filename_hint: str = "audio.webm") -> str:
    """Send audio bytes to OpenAI Whisper (or compatible) and return transcript text."""
    client = get_client()
    if not client:
        return "[No API key] Please add your OpenAI API key."

    # Save bytes to a temp file so the API can read it
    suffix = "." + filename_hint.split(".")[-1].lower() if "." in filename_hint else ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL,
                file=f
            )
        # SDK returns text under .text for whisper-1 and gpt-4o-mini-transcribe
        text = getattr(resp, "text", None)
        if not text:
            try:
                text = resp["text"]  # some servers return dict-like
            except Exception:
                text = ""
        return text or "[Transcription empty]"
    except Exception as e:
        return f"[Transcription error] {e}"
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def robust_json_extract(s: str) -> dict:
    """Extract the first JSON object in a string (handles code fences)."""
    import re
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def rate_and_feedback(question: str, subject: str, transcript: str) -> dict:
    """Ask the LLM to rate answer (1-10) + give explanation & feedback; return dict."""
    client = get_client()
    if not client:
        # Heuristic fallback if no API
        word_count = len(transcript.split())
        base = 5
        if word_count < 30:
            base -= 2
        if "result" in transcript.lower() or "improve" in transcript.lower():
            base += 1
        base = max(1, min(10, base))
        return {
            "rating_out_of_10": base,
            "explanation": "Heuristic: based on length and presence of impact/result language.",
            "feedback": "Use STAR (Situation, Task, Action, Result). Add concrete metrics to strengthen your answer.",
            "strengths": "Relevant and understandable.",
            "improvements": "Add numbers, reduce filler, end with a clear result.",
            "sample_answer": "Start with context, explain your specific actions, quantify the outcome, and reflect briefly."
        }

    system = (
        "You are an expert interview coach for freshers. "
        "Evaluate the candidate's answer concisely and kindly. "
        "OUTPUT STRICT JSON with keys: rating_out_of_10 (integer 1-10), explanation, feedback, strengths, improvements, sample_answer."
    )
    user = (
        f"Subject: {subject}\n"
        f"Question: {question}\n\n"
        f"Candidate answer (transcript):\n{transcript}\n\n"
        "Rate the answer from 1-10 and explain. Provide concise strengths and improvements.\n"
        "Return ONLY a JSON object with the required keys."
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content
        data = robust_json_extract(content) or {}
        # Coerce rating to int if possible
        try:
            data["rating_out_of_10"] = int(data.get("rating_out_of_10", 0))
        except Exception:
            data["rating_out_of_10"] = 0
        return data
    except Exception as e:
        return {
            "rating_out_of_10": 0,
            "explanation": f"[LLM error] {e}",
            "feedback": "Try again or check API key/model.",
            "strengths": "",
            "improvements": "",
            "sample_answer": ""
        }

# ---------- Session state ----------
if "mode" not in st.session_state:
    st.session_state.mode = "Manual Mode"
if "subject" not in st.session_state:
    st.session_state.subject = list(QUESTION_BANK.keys())[0]
if "q_index" not in st.session_state:
    st.session_state.q_index = 0
if "records" not in st.session_state:
    st.session_state.records = []
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""
if "practice_questions" not in st.session_state:
    st.session_state.practice_questions = []
if "practice_current" not in st.session_state:
    st.session_state.practice_current = 0

# ---------- Mode selector ----------
st.session_state.mode = st.radio("Select Mode", ["Manual Mode", "Practice Set Mode"], horizontal=True)

# ---------- Global reset ----------
if st.button("Reset Session"):
    st.session_state.mode = "Manual Mode"
    st.session_state.subject = list(QUESTION_BANK.keys())[0]
    st.session_state.q_index = 0
    st.session_state.records = []
    st.session_state.transcript_text = ""
    st.session_state.practice_questions = []
    st.session_state.practice_current = 0
    st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

# ======================= MANUAL MODE =======================
if st.session_state.mode == "Manual Mode":
    col_left, col_right = st.columns([4, 2])

    # ---------- UI: Select subject & question ----------
    with col_left:
        st.subheader("Step 1: Choose Subject & Question")
        st.session_state.subject = st.selectbox("Subject to practice", list(QUESTION_BANK.keys()), index=0)
        questions = QUESTION_BANK[st.session_state.subject]
        q = questions[st.session_state.q_index % len(questions)]
        st.markdown(f"**Question:** {q}")

        c1, c2 = st.columns(2)
        if c1.button("⬅ Previous"):
            st.session_state.q_index = (st.session_state.q_index - 1) % len(questions)
            st.session_state.transcript_text = ""
            st.rerun()
        if c2.button("Next ➡"):
            st.session_state.q_index = (st.session_state.q_index + 1) % len(questions)
            st.session_state.transcript_text = ""
            st.rerun()

    # ---------- UI: Record / upload audio ----------
    with col_left:
        st.subheader("Step 2: Record or Upload Your Answer (Audio)")
        st.write("Click below to capture audio from your mic (or upload an audio file). Then press **Transcribe**.")
        audio_file = st.audio_input("🎙️ Record / Upload", help="Record your answer with your microphone, or upload a .wav/.mp3/.webm file.")
        if audio_file is not None:
            st.audio(audio_file)

        if st.button("📝 Transcribe Audio to Text"):
            if audio_file is None:
                st.warning("Please record or upload an audio answer first.")
            else:
                try:
                    audio_bytes = audio_file.read()
                except Exception:
                    audio_bytes = audio_file.getvalue()
                transcript = transcribe_audio_filebytes(audio_bytes, filename_hint=audio_file.name)
                st.session_state.transcript_text = transcript

    # ---------- UI: Edit transcript ----------
    with col_left:
        st.subheader("Step 3: Review / Edit Your Transcript")
        st.session_state.transcript_text = st.text_area(
            "Transcript (you can edit before submitting)",
            value=st.session_state.transcript_text or "",
            height=180,
            placeholder="Your transcribed answer will appear here..."
        )

    # ---------- Submit for rating ----------
    with col_left:
        st.subheader("Step 4: Submit & Get Rating + Feedback")
        if st.button("✅ Submit Answer"):
            if not st.session_state.transcript_text.strip():
                st.warning("Please record/transcribe your answer (or type it) before submitting.")
            else:
                with st.spinner("Scoring your answer..."):
                    result = rate_and_feedback(
                        question=QUESTION_BANK[st.session_state.subject][st.session_state.q_index % len(questions)],
                        subject=st.session_state.subject,
                        transcript=st.session_state.transcript_text.strip()
                    )

                # Normalize result keys
                rating = result.get("rating_out_of_10", 0)
                explanation = result.get("explanation", "")
                feedback = result.get("feedback", "")
                strengths = result.get("strengths", "")
                improvements = result.get("improvements", "")
                sample_answer = result.get("sample_answer", "")

                # 🔥 Show everything immediately (Rating + Explanation + Feedback + Sample Answer)
                st.success(f"Rating: **{rating}/10**")
                with st.expander("Explanation", expanded=True):
                    st.write(explanation)
                with st.expander("Feedback", expanded=True):
                    st.write(feedback)
                col_s, col_i = st.columns(2)
                with col_s:
                    st.markdown("**Strengths**")
                    st.write(strengths)
                with col_i:
                    st.markdown("**Improvements**")
                    st.write(improvements)
                with st.expander("Sample Answer"):
                    st.write(sample_answer)

                # Save record
                rec = QARecord(
                    subject=st.session_state.subject,
                    question=QUESTION_BANK[st.session_state.subject][st.session_state.q_index % len(questions)],
                    transcript=st.session_state.transcript_text.strip(),
                    rating_10=int(rating) if isinstance(rating, int) else 0,
                    explanation=explanation,
                    feedback=feedback,
                    strengths=strengths,
                    improvements=improvements,
                    sample_answer=sample_answer
                )
                st.session_state.records.append(rec)

    # ---------- Right side: history & export ----------
    with col_right:
        st.subheader("Step 5: History & Download")
        if st.session_state.records:
            avg = sum(r.rating_10 for r in st.session_state.records) / len(st.session_state.records)
            st.metric("Average Rating", round(avg, 2))
            st.progress(min(1.0, max(0.0, avg/10)))

            for i, r in enumerate(reversed(st.session_state.records), start=1):
                with st.expander(f"#{len(st.session_state.records)-i+1} — {r.subject}: {r.question[:60]}"):
                    st.write(f"**Rating:** {r.rating_10}/10")
                    st.write(f"**Transcript:** {r.transcript}")
                    st.write(f"**Strengths:** {r.strengths}")
                    st.write(f"**Improvements:** {r.improvements}")
                    st.write(f"**Feedback:** {r.feedback}")

            # Download JSON
            as_json = json.dumps([asdict(r) for r in st.session_state.records], indent=2, ensure_ascii=False)
            st.download_button(
                "⬇️ Download Session (JSON)",
                data=as_json.encode("utf-8"),
                file_name="interview_session.json",
                mime="application/json",
                use_container_width=True
            )

    st.markdown("---")
    st.caption("Tip: If you don't see the mic recorder, upgrade Streamlit. Use the sidebar or secrets to keep your API key safe.")

# =================== PRACTICE SET MODE ===================
elif st.session_state.mode == "Practice Set Mode":
    # Setup screen (pick subject + number of questions)
    if not st.session_state.practice_questions:
        st.session_state.subject = st.selectbox("Select Subject", list(QUESTION_BANK.keys()), index=0)
        num_q = st.number_input("Number of questions", min_value=1, max_value=len(QUESTION_BANK[st.session_state.subject]), value=3)
        if st.button("Start Practice Set"):
            st.session_state.practice_questions = random.sample(QUESTION_BANK[st.session_state.subject], num_q)
            st.session_state.practice_current = 0
            st.session_state.records = []
            st.rerun()

    # Running the set
    if st.session_state.practice_questions and st.session_state.practice_current < len(st.session_state.practice_questions):
        total = len(st.session_state.practice_questions)
        idx = st.session_state.practice_current
        q = st.session_state.practice_questions[idx]

        st.subheader(f"Question {idx+1} of {total}")
        st.progress((idx) / total)
        st.write(q)

        audio_file = st.audio_input("🎙️ Record / Upload Answer")
        if audio_file and st.button("📝 Transcribe Audio"):
            audio_bytes = audio_file.read()
            st.session_state.transcript_text = transcribe_audio_filebytes(audio_bytes, filename_hint=audio_file.name)

        st.session_state.transcript_text = st.text_area("Transcript", value=st.session_state.transcript_text, height=150)

        if st.button("✅ Submit Answer (Next)"):
            result = rate_and_feedback(q, st.session_state.subject, st.session_state.transcript_text.strip())
            rec = QARecord(
                subject=st.session_state.subject,
                question=q,
                transcript=st.session_state.transcript_text.strip(),
                rating_10=int(result.get("rating_out_of_10", 0)),
                explanation=result.get("explanation", ""),
                feedback=result.get("feedback", ""),
                strengths=result.get("strengths", ""),
                improvements=result.get("improvements", ""),
                sample_answer=result.get("sample_answer", "")
            )
            st.session_state.records.append(rec)
            st.session_state.practice_current += 1
            st.session_state.transcript_text = ""
            st.rerun()

    # Completed summary
    if st.session_state.practice_questions and st.session_state.practice_current >= len(st.session_state.practice_questions):
        st.success("✅ Practice Set Completed!")
        avg = sum(r.rating_10 for r in st.session_state.records) / len(st.session_state.records)
        st.metric("Average Rating", round(avg, 2))
        st.progress(1.0)

        for r in st.session_state.records:
            with st.expander(f"{r.question} — {r.rating_10}/10"):
                st.write(f"**Transcript:** {r.transcript}")
                st.write(f"**Strengths:** {r.strengths}")
                st.write(f"**Improvements:** {r.improvements}")
                st.write(f"**Feedback:** {r.feedback}")
                st.write(f"**Sample Answer:** {r.sample_answer}")

        as_json = json.dumps([asdict(r) for r in st.session_state.records], indent=2)
        st.download_button("⬇️ Download Report", data=as_json.encode(), file_name="practice_set_report.json", mime="application/json", use_container_width=True)
