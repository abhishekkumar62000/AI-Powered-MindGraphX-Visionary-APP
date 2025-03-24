import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
import streamlit.components.v1 as components
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
from textblob import TextBlob
from xhtml2pdf import pisa

# Download NLTK resources only once
nltk.data.path.append("nltk_data")  # Ensure a local download path
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

load_dotenv()

API_KEY = os.getenv('API_KEY')

def configure_genai():
    """Configure the Gemini AI with the API key."""
    if not API_KEY:
        st.error("API Key is missing. Please provide a valid Google API key.")
        return False
    try:
        genai.configure(api_key=API_KEY)
        return True
    except Exception as e:
        st.error(f"Error configuring Google API: {str(e)}")
        return False

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            st.warning("No text could be extracted from the PDF. It might be scanned or image-based.")
            return None
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def preprocess_text(text):
    """Preprocess the text by removing stop words."""
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
    return filtered_text

def generate_summary(text):
    """
    Generate a summary of the text using Gemini AI, with chunking if needed.
    Debug prints are added to see the response for each chunk.
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        max_chunk_size = 50000  # Increased chunk size for large PDFs

        if len(text) > max_chunk_size:
            st.write("DEBUG: Splitting text into chunks for summary...")
            # Split text into chunks
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            summaries = []

            for idx, chunk in enumerate(chunks):
                prompt = f"Summarize the following text in 5-7 bullet points:\n\n{chunk}"
                response = model.generate_content(prompt)
                st.write(f"DEBUG: Summary Chunk {idx+1} length={len(chunk)}")
                st.write(f"DEBUG: Summary Chunk {idx+1} AI response:", response.text)

                if response.text and response.text.strip():
                    summaries.append(response.text.strip())
                else:
                    summaries.append(f"⚠ Could not generate summary for chunk {idx+1}.")

            # Combine chunk summaries
            combined_summary = "\n".join(summaries)

            # Re-summarize the combined summary
            final_prompt = f"Summarize the following bullet points into 5-7 concise bullet points:\n\n{combined_summary}"
            final_response = model.generate_content(final_prompt)

            st.write("DEBUG: Final combined summary response:", final_response.text)

            if final_response.text and final_response.text.strip():
                return final_response.text.strip()
            else:
                return combined_summary

        else:
            # No chunking needed
            prompt = f"Summarize the following text in 5-7 bullet points:\n\n{text}"
            response = model.generate_content(prompt)
            st.write("DEBUG: Single-chunk summary response:", response.text)

            if not response.text or not response.text.strip():
                return "⚠ Could not generate summary."
            return response.text.strip()

    except Exception as e:
        return f"Error generating summary: {str(e)}"

def create_mindmap_markdown(text):
    """
    Create a hierarchical markdown mindmap from the text using Gemini AI, with chunking if needed.
    Debug prints are added to see the response for each chunk.
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        max_chunk_size = 50000  # Increased chunk size for large PDFs

        if len(text) > max_chunk_size:
            st.write("DEBUG: Splitting text into chunks for mindmap...")
            # Split text into chunks
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            mindmaps = []

            for idx, chunk in enumerate(chunks):
                prompt = """
                Create a hierarchical markdown mindmap from the following text.
                Use proper markdown heading syntax (# for main topics, ## for subtopics, ### for details).
                Focus on the main concepts and their relationships.
                Include relevant details and connections between ideas.
                Keep the structure clean and organized.
                
                Format the output exactly like this example:
                # Main Topic
                ## Subtopic 1
                ### Detail 1
                - Key point 1
                - Key point 2
                ### Detail 2
                ## Subtopic 2
                ### Detail 3
                ### Detail 4
                
                Text to analyze: {text}
                
                Respond only with the markdown mindmap, no additional text.
                """
                response = model.generate_content(prompt.format(text=chunk))
                st.write(f"DEBUG: Mindmap Chunk {idx+1} length={len(chunk)}")
                st.write(f"DEBUG: Mindmap Chunk {idx+1} AI response:", response.text)

                if response.text and response.text.strip():
                    mindmaps.append(f"# Chunk {idx+1}\n" + response.text.strip())
                else:
                    mindmaps.append(f"# Chunk {idx+1}\n⚠ Could not generate mindmap for chunk {idx+1}.")

            # Combine chunk mindmaps into a single markdown
            combined_mindmap = "\n\n".join(mindmaps)
            st.write("DEBUG: Final combined mindmap markdown:")
            st.write(combined_mindmap)
            return combined_mindmap

        else:
            # No chunking needed
            prompt = """
            Create a hierarchical markdown mindmap from the following text.
            Use proper markdown heading syntax (# for main topics, ## for subtopics, ### for details).
            Focus on the main concepts and their relationships.
            Include relevant details and connections between ideas.
            Keep the structure clean and organized.
            
            Format the output exactly like this example:
            # Main Topic
            ## Subtopic 1
            ### Detail 1
            - Key point 1
            - Key point 2
            ### Detail 2
            ## Subtopic 2
            ### Detail 3
            ### Detail 4
            
            Text to analyze: {text}
            
            Respond only with the markdown mindmap, no additional text.
            """
            response = model.generate_content(prompt.format(text=text))
            st.write("DEBUG: Single-chunk mindmap AI response:", response.text)

            if not response.text or not response.text.strip():
                st.error("Received empty response from Gemini AI for mindmap.")
                return None
            return response.text.strip()

    except Exception as e:
        st.error(f"Error generating mindmap: {str(e)}")
        return None

def create_markmap_html(markdown_content):
    """
    Create HTML with enhanced Markmap visualization, stylish design & interactive features.
    """
    # Safely escape backticks and dollar braces
    markdown_content = markdown_content.replace('`', '\\`').replace('${', '\\${}')

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            #mindmap-container {{
                width: 100%;
                height: 800px;
                margin: 20px auto;
                padding: 10px;
                background: #f4f4f9;
                border: 2px solid #ddd;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            #mindmap {{
                width: 90%;
                height: 90%;
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/d3@6"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-view"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-lib@0.14.3/dist/browser/index.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.4.0/jspdf.umd.min.js"></script>
    </head>
    <body>
        <div id="mindmap-container">
            <svg id="mindmap"></svg>
        </div>
        <button id="download-mindmap">Download Mindmap as PDF</button>

        <script>
            window.onload = async () => {{
                try {{
                    const markdown = `{markdown_content}`;
                    const transformer = new markmap.Transformer();
                    const {{ root }} = transformer.transform(markdown);
                    const mm = new markmap.Markmap(document.querySelector('#mindmap'), {{
                        maxWidth: 800,
                        color: (node) => {{
                            const level = node.depth;
                            return ['#2196f3', '#4caf50', '#ff9800', '#f44336'][level % 4];
                        }},
                        paddingX: 16,
                        autoFit: true,
                        initialExpandLevel: 2,
                        duration: 500,
                    }});
                    mm.setData(root);
                    mm.fit();
                }} catch (error) {{
                    console.error('Error rendering mindmap:', error);
                    document.body.innerHTML = '<p style="color: red;">Error rendering mindmap. Please check the console for details.</p>';
                }}
            }};
            
            document.getElementById('download-mindmap').addEventListener('click', function() {{
                setTimeout(() => {{
                    let mindmapElement = document.getElementById('mindmap-container');
                    html2canvas(mindmapElement, {{
                        scale: 2,  // Higher resolution
                        useCORS: true
                    }}).then(canvas => {{
                        let imgData = canvas.toDataURL('image/png');
                        let {{ jsPDF }} = window.jspdf;
                        let pdf = new jsPDF('l', 'mm', 'a4');
                        pdf.addImage(imgData, 'PNG', 10, 10, 280, 150);
                        pdf.save("mindmap.pdf");
                    }});
                }}, 1000); // Delay to allow rendering
            }});
        </script>
    </body>
    </html>
    """
    return html_content


def save_mindmap_as_image(html_content):
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get("data:text/html;charset=utf-8," + html_content)
    time.sleep(2)  # Add a delay to ensure the content is fully rendered
    driver.save_screenshot("mindmap.png")
    driver.quit()

def save_mindmap_as_pdf(html_content):
    pdf_file = "mindmap.pdf"
    with open(pdf_file, "wb") as pdf:
        pisa_status = pisa.CreatePDF(html_content, dest=pdf)
    return pdf_file if not pisa_status.err else None

def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "😊 Positive"
    elif sentiment < 0:
        return "😞 Negative"
    else:
        return "😐 Neutral"

def main():
    st.set_page_config(layout="wide")
    
    st.title("🚨AI Powered MindGraphX Visionary APP💡🧠")
    st.caption("AI-powered deep brain connections for mind mapping.🧠")
    
    AI_path = "AI.png"  # Ensure this file is in the same directory as your script
    try:
        st.sidebar.image(AI_path)
    except FileNotFoundError:
        st.sidebar.warning("AI.png file not found. Please check the file path.")
        
    image_path = "image.png"  # Ensure this file is in the same directory as your script
    try:
        st.sidebar.image(image_path)
    except FileNotFoundError:
        st.sidebar.warning("image.png file not found. Please check the file path.")
        
    st.sidebar.markdown("👨‍💻Developer: Abhishek❤️Yadav")
    developer_path = "pic.jpg"  # Ensure this file is in the same directory as your script
    try:
        st.sidebar.image(developer_path)
    except FileNotFoundError:
        st.sidebar.warning("pic.jpg file not found. Please check the file path.")
    
    if not configure_genai():
        return

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("🚨Processing PDF and Generating Mindmap🧠...."):
            text = extract_text_from_pdf(uploaded_file)
            
            if text:
                st.info(f"Successfully extracted {len(text)} characters from PDF")
                
                if st.checkbox("Preprocess text before generating mindmap"):
                    text = preprocess_text(text)
                    st.write("DEBUG: Preprocessed text:", text)
                
                # Perform Sentiment Analysis
                if st.checkbox("📌 🧠Perform Sentiment Analysis😊"):
                    sentiment_result = analyze_sentiment(text)
                    st.write(f"📊 Sentiment Analysis Result: {sentiment_result}")
                
                # Generate and display AI summary (with chunking if needed)
                summary = generate_summary(text)
                st.subheader("📌 AI-Generated Summary🧠")
                st.write(summary)
                
                # Generate mindmap markdown (with chunking if needed)
                markdown_content = create_mindmap_markdown(text)

                st.write("DEBUG: Final Mindmap Markdown:")
                st.write(markdown_content)

                if markdown_content:
                    tab1, tab2 = st.tabs(["📊 Mindmap🤷‍♂️", "📝 Markdown & Export🤷‍♂️"])
                    
                    with tab1:
                        st.subheader("Interactive Mindmap")
                        html_content = create_markmap_html(markdown_content)
                        components.html(html_content, height=850, scrolling=True)
                        
                        # MindMap section
                        st.subheader("📥 Download MindMap")
                        if st.button("📥 Download Mindmap as PNG"):
                            save_mindmap_as_image(html_content)
                            with open("mindmap.png", "rb") as img_file:
                                st.download_button("Download Image", img_file, "mindmap.png", "image/png")
                        
                        # Add button to download mindmap as PDF
                        st.markdown("""
                            <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.4.0/jspdf.umd.min.js"></script>
                            <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/0.5.0/html2canvas.min.js"></script>

                            <button id="download-mindmap">Download Mindmap as PDF</button>

                            <script>
                                document.getElementById('download-mindmap').addEventListener('click', function() {
                                    let mindmap = document.getElementById('mindmap-container');
                                    html2canvas(mindmap).then(canvas => {
                                        let imgData = canvas.toDataURL('image/png');
                                        let { jsPDF } = window.jspdf;
                                        let pdf = new jsPDF('p', 'mm', 'a4');
                                        pdf.addImage(imgData, 'PNG', 10, 10, 180, 100);
                                        pdf.save("mindmap.pdf");
                                    });
                                });
                            </script>
                        """, unsafe_allow_html=True)
                    
                    with tab2:
                        st.subheader("Generated Markdown")
                        st.text_area("Markdown Content", markdown_content, height=400)
                        
                        # Prepare data for export
                        json_data = json.dumps({"mindmap": markdown_content}, indent=4)
                        csv_data = pd.DataFrame([{"Markdown": markdown_content}]).to_csv(index=False)
                        
                        st.download_button("⬇ Download Markdown", markdown_content, "mindmap.md", "text/markdown")
                        st.download_button("⬇ Download JSON", json_data, "mindmap.json", "application/json")
                        st.download_button("⬇ Download CSV", csv_data, "mindmap.csv", "text/csv")
                        
                else:
                    st.error("Could not generate mindmap content. Gemini AI might have returned an empty response.")
            else:
                st.error("No text extracted from the PDF. It might be scanned or not text-based.")

    st.write("---")
    st.write("### (Optional) 👨‍💻Test Mindmap Without PDF🤷‍♂️")
    test_text = st.text_area("Enter some sample text to create a mindmap:", "This is a sample text to test mindmap.")
    if st.button("Generate Mindmap from Text"):
        st.write("DEBUG: Using test_text for mindmap.")
        test_markdown = create_mindmap_markdown(test_text)
        if test_markdown:
            st.write("Markdown Generated:", test_markdown)
            html_content = create_markmap_html(test_markdown)
            components.html(html_content, height=850, scrolling=True)
        else:
            st.error("Failed to generate mindmap from test text.")

if __name__ == "__main__":
    main()
