from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
import wandb
import PyPDF2
import io
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

app = FastAPI()

# Load environment variables and initialize WandB
load_dotenv()
print("Initializing WandB...")
wandb.init(project="Visibility Example - AIE3", entity="tehnickapodrska")
openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key

class RAGSystem:
    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = None

    def ingest_document(self, text, source):
        if text.strip():
            self.documents.append({"text": text, "source": source})
            self._update_vectors()

    def _update_vectors(self):
        corpus = [doc['text'] for doc in self.documents]
        if corpus:
            self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query):
        if self.doc_vectors is None or self.doc_vectors.shape[0] == 0:
            raise ValueError("Document corpus is empty.")
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        if similarities.size == 0:
            raise ValueError("No similarities calculated. Check the input and model fitting.")
        best_match_index = similarities.argmax()
        return self.documents[best_match_index]

rag_system = RAGSystem()

async def extract_text_from_pdf(file: UploadFile):
    contents = await file.read()
    pdf_stream = io.BytesIO(contents)
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text

@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile] = File(...)):
    for file in files:
        if file.filename.endswith('.pdf'):
            text = await extract_text_from_pdf(file)
            if text.strip():
                rag_system.ingest_document(text, source=file.filename)
    return {"message": "Files processed successfully."}

@app.post("/query/")
async def handle_query(query: str = Form(...)):
    try:
        document = rag_system.retrieve(query)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Here is the document text:\n\n{document['text']}\n\nQuestion: {query}"}
            ]
        )
        answer = response.choices[0].message.content
        response_content = f"<html><body><h3>Best document source: {document['source']}</h3>" \
                           f"<p>OpenAI Response: {answer}</p>" \
                           f"<a href='https://wandb.ai/your_username/your_project'>View on WandB</a></body></html>"
        return HTMLResponse(content=response_content)
    except Exception as e:
        return HTMLResponse(content=f"<html><body><h3>Error: {str(e)}</h3></body></html>", status_code=400)

@app.get("/")
async def main():
    content = """
<body>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit" value="Upload PDFs">
</form>
<br>
<form action="/query/" method="post">
<input name="query" type="text" placeholder="Enter your question here...">
<input type="submit" value="Submit Query">
</form>
</body>
    """
    return HTMLResponse(content=content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
