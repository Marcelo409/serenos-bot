import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

CARPETA_DOCS = "documentos"
CARPETA_DB = "chroma_db"

print("Inicializando modelo de embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    encode_kwargs={"normalize_embeddings": True}
)

# Si la base vectorial ya existe en disco, la cargamos. Si no, la construimos.
if Path(CARPETA_DB).exists() and any(Path(CARPETA_DB).iterdir()):
    print("Base vectorial encontrada en disco. Cargando...")
    db = Chroma(persist_directory=CARPETA_DB, embedding_function=embeddings)
    print(f"Base cargada con {db._collection.count()} chunks.")
else:
    print("No hay base vectorial en disco. Construyendo desde cero...")

    print("Cargando documentos...")
    loader = PyPDFDirectoryLoader(f"{CARPETA_DOCS}/")
    docs = loader.load()
    print(f"Cargadas {len(docs)} paginas")

    print("Partiendo en chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Generados {len(chunks)} chunks")

    print("Calculando embeddings y guardando en disco (esto tarda 3-5 minutos la primera vez)...")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CARPETA_DB
    )
    print("Base vectorial guardada. Las proximas ejecuciones seran rapidas.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

PROMPT_TEMPLATE = """Eres un asistente para serenos de la municipalidad de Chorrillos.
Tu tarea es responder preguntas usando el siguiente contexto extraido de documentos oficiales.

Reglas:
- Si la respuesta esta directamente en el contexto, respondela citando la pagina.
- Si el contexto menciona el tema pero no responde completamente, resume lo que si dice y aclara que la respuesta es parcial.
- Solo si el contexto no menciona el tema en absoluto, responde: "No tengo esa informacion en los documentos disponibles."
- No inventes datos, nombres, cifras ni procedimientos.

Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:"""

print("\nListo. Hazme preguntas (escribe salir para terminar).\n")

while True:
    pregunta = input("Pregunta: ")
    if pregunta.lower() in ["salir", "exit", "quit"]:
        break

    relevantes = db.similarity_search(pregunta, k=6)

    bloques = []
    for d in relevantes:
        pagina = d.metadata.get("page", "?")
        bloques.append(f"[Pagina {pagina}] {d.page_content}")
    contexto = "\n\n".join(bloques)

    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pregunta=pregunta)
    respuesta = llm.invoke(prompt)

    print("\nRespuesta:", respuesta.content)
    paginas = sorted(set([d.metadata.get("page", "?") for d in relevantes]))
    fuentes = sorted(set([os.path.basename(d.metadata.get("source", "?")) for d in relevantes]))
    print(f"Fuentes: {fuentes} | Paginas: {paginas}\n")