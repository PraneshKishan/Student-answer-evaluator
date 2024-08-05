import streamlit as st
import PyPDF2
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from io import BytesIO

# Load NLP model
nlp = spacy.load('en_core_web_sm')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to evaluate answers
def evaluate_answer(student_answer, reference_text):
    student_doc = nlp(student_answer)
    reference_doc = nlp(reference_text)
    
    student_sentences = list(student_doc.sents)
    reference_sentences = list(reference_doc.sents)
    
    feedback = []
    similarity_scores = []
    
    similarity_threshold = 0.9  # Adjusted threshold
    
    for i, sent in enumerate(student_sentences):
        if i < len(reference_sentences):
            similarity = sent.similarity(reference_sentences[i])
            similarity_scores.append(similarity)
            
            if similarity < similarity_threshold:
                feedback.append({
                    "sentence": sent.text,
                    "reference": reference_sentences[i].text,
                    "similarity": similarity
                })
    
    return feedback, similarity_scores

# Function to generate feedback paragraph and analytics
def generate_feedback_paragraph(feedback):
    if not feedback:
        return "Your answer is quite similar to the source material. No major revisions are needed."
    
    feedback_paragraph = "Here's a detailed feedback on your answer:\n\n"
    for item in feedback:
        feedback_paragraph += (
            f"Sentence: \"{item['sentence']}\"\n"
            f"Reference: \"{item['reference']}\"\n"
            f"Similarity: {item['similarity']:.2f}\n\n"
            "Your answer might benefit from the following improvements:\n"
            "1. Ensure that each sentence closely aligns with the key points in the reference text.\n"
            "2. Pay attention to specific terminology used in the source material.\n"
            "3. Correct any factual inaccuracies or misunderstandings.\n\n"
        )
    
    return feedback_paragraph

def generate_analytics(feedback, similarity_scores):
    if not similarity_scores:
        return {"average_similarity": 0.0, "mistake_percentage": 0.0, "plot": None}
    
    df = pd.DataFrame(feedback)
    average_similarity = pd.Series(similarity_scores).mean()
    total_sentences = len(similarity_scores)
    mistakes_count = len(feedback)
    mistake_percentage = (mistakes_count / total_sentences) * 100 if total_sentences > 0 else 0

    # Create a plot for similarity scores
    plt.figure(figsize=(12, 6))
    sns.histplot(similarity_scores, kde=True)
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    plot_img = buf.getvalue()
    buf.close()
    
    return {
        "average_similarity": average_similarity,
        "mistake_percentage": mistake_percentage,
        "plot": plot_img
    }

# Streamlit app
st.title("Intelligent Answer Evaluation System")

uploaded_source_pdf = st.file_uploader("Upload Source PDF", type="pdf")
uploaded_answer_pdf = st.file_uploader("Upload Answer PDF", type="pdf")

if uploaded_source_pdf and uploaded_answer_pdf:
    source_text = extract_text_from_pdf(uploaded_source_pdf)
    answer_text = extract_text_from_pdf(uploaded_answer_pdf)

    feedback, similarity_scores = evaluate_answer(answer_text, source_text)
    
    st.write("Feedback:")
    feedback_paragraph = generate_feedback_paragraph(feedback)
    st.write(feedback_paragraph)

    analytics = generate_analytics(feedback, similarity_scores)

    st.write("Analytics:")
    st.write(f"Average Similarity: {analytics['average_similarity']:.2f}")
    st.write(f"Mistake Percentage: {analytics['mistake_percentage']:.2f}%")
    
    if analytics['plot']:
        st.image(analytics['plot'], caption='Distribution of Similarity Scores')
