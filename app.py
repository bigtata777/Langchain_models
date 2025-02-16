from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma  # Usamos Chroma en lugar de FAISS
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableSequence
# Cargar variables de entorno
load_dotenv()

# Inicializar FastAPI
app = FastAPI()

# Montar la carpeta static para servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configurar Jinja2 para manejar plantillas HTML
templates = Jinja2Templates(directory="templates")

# Inicializar el modelo de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)


# Ajustar la ruta del PDF a una ruta relativa
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "pdfs", "08_tratamiento_asma.pdf")

# Verificar si el archivo existe antes de cargarlo
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"El archivo PDF no se encontró en: {pdf_path}")

print(f"📄 Cargando PDF desde: {pdf_path}")  # Depuración en logs de Render

# Cargar y procesar el PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Dividir el texto en fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Crear embeddings y almacenarlos en Chroma
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma.from_documents(texts, embeddings)  # Usamos Chroma

# Definir un prompt template
prompt = PromptTemplate(
    input_variables=["tema", "contexto"],
    template="Basado en el siguiente contexto, responde la pregunta de manera clara y concisa:\n\nContexto:\n{contexto}\n\nPregunta:\n{tema}\n\nRespuesta:"
)

# Crear la cadena con el modelo y el prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Ruta para la página principal
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Ruta para manejar la consulta del usuario
@app.post("/consulta", response_class=HTMLResponse)
async def procesar_consulta(request: Request, tema: str = Form(...)):
    # Buscar fragmentos relevantes en el PDF
    docs = vectorstore.similarity_search(tema, k=2)  # Buscar los 2 fragmentos más relevantes
    contexto = "\n".join([doc.page_content for doc in docs])  # Combinar los fragmentos en un solo contexto

    # Ejecutar la cadena con el tema y el contexto
    respuesta = chain.run(tema=tema, contexto=contexto)

    # Devolver la respuesta al frontend
    return templates.TemplateResponse("index.html", {"request": request, "respuesta": respuesta})
