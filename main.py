import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import streamlit.components.v1 as components
import json
import pandas as pd

# Define the API key directly in the code
API_KEY = "AIzaSyB5K03NQmrg38rla3Dh1sNLJ_wvCh2bYHU"

def configure_genai():
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
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            st.warning("No text could be extracted from the PDF. Please ensure it's not scanned or image-based.")
            return None
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def generate_summary(text):
    try:
        model = genai.GenerativeModel('gemini-pro')
        max_chunk_size = 50000  # Increased chunk size
        if len(text) > max_chunk_size:
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            summaries = []
            for chunk in chunks:
                prompt = f"Summarize the following text in 5-7 bullet points:\n\n{chunk}"
                response = model.generate_content(prompt)
                if response.text and response.text.strip():
                    summaries.append(response.text.strip())
                else:
                    summaries.append("⚠️ Could not generate summary for this chunk.")
            combined_summary = "\n".join(summaries)
            final_prompt = f"Summarize the following bullet points into 5-7 concise bullet points:\n\n{combined_summary}"
            final_response = model.generate_content(final_prompt)
            if final_response.text and final_response.text.strip():
                return final_response.text.strip()
            else:
                return combined_summary
        else:
            prompt = f"Summarize the following text in 5-7 bullet points:\n\n{text}"
            response = model.generate_content(prompt)
            if not response.text or not response.text.strip():
                return "⚠️ Could not generate summary."
            return response.text.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def create_mindmap_markdown(text):
    try:
        model = genai.GenerativeModel('gemini-pro')
        max_chunk_size = 50000  # Increased chunk size
        if len(text) > max_chunk_size:
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
                if response.text and response.text.strip():
                    mindmaps.append(f"# Chunk {idx+1}\n" + response.text.strip())
                else:
                    mindmaps.append(f"# Chunk {idx+1}\n⚠️ Could not generate mindmap for this chunk.")
            combined_mindmap = "\n\n".join(mindmaps)
            return combined_mindmap
        else:
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
            if not response.text or not response.text.strip():
                st.error("Received empty response from Gemini AI")
                return None
            return response.text.strip()
    except Exception as e:
        st.error(f"Error generating mindmap: {str(e)}")
        return None

def create_markmap_html(markdown_content):
    markdown_content = markdown_content.replace('`', '\\`').replace('${', '\\${')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            #mindmap {{
                width: 100%;
                height: 800px;
                margin: 20px auto;
                padding: 10px;
                background: #f4f4f9;
                border: 2px solid #ddd;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/d3@6"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-view"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-lib@0.14.3/dist/browser/index.min.js"></script>
    </head>
    <body>
        <svg id="mindmap"></svg>
        <script>
            window.onload = async () => {{
                try {{
                    const markdown = `{markdown_content}`;
                    const transformer = new markmap.Transformer();
                    const {{root}} = transformer.transform(markdown);
                    const mm = new markmap.Markmap(document.querySelector('#mindmap'), {{
                        maxWidth: 300,
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
        </script>
    </body>
    </html>
    """
    return html_content

def main():
    st.set_page_config(layout="wide")
    
    st.title("📚AI PDF to Interactive Mindmap Converter ")
    
    if not configure_genai():
        return

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("🔄 Processing PDF and generating mindmap..."):
            text = extract_text_from_pdf(uploaded_file)
            
            if text:
                st.info(f"Successfully extracted {len(text)} characters from PDF")
                
                # Generate and display AI summary with chunking if needed
                summary = generate_summary(text)
                st.subheader("📌 AI-Generated Summary")
                st.write(summary)
                
                # Generate mindmap markdown using chunking if necessary
                markdown_content = create_mindmap_markdown(text)
                
                if markdown_content:
                    tab1, tab2 = st.tabs(["📊 Mindmap", "📝 Markdown & Export"])
                    
                    with tab1:
                        st.subheader("Interactive Mindmap")
                        html_content = create_markmap_html(markdown_content)
                        components.html(html_content, height=800, scrolling=True)
                    
                    with tab2:
                        st.subheader("Generated Markdown")
                        st.text_area("Markdown Content", markdown_content, height=400)
                        
                        json_data = json.dumps({"mindmap": markdown_content}, indent=4)
                        csv_data = pd.DataFrame([{ "Markdown": markdown_content }]).to_csv(index=False)
                        
                        st.download_button("⬇️ Download Markdown", markdown_content, "mindmap.md", "text/markdown")
                        st.download_button("⬇️ Download JSON", json_data, "mindmap.json", "application/json")
                        st.download_button("⬇️ Download CSV", csv_data, "mindmap.csv", "text/csv")

if __name__ == "__main__":
    main()
