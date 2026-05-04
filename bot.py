import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print("Cargando documentos...")
loader = PyPDFDirectoryLoader("documentos/")
docs = loader.load()
print(f"Cargadas {len(docs)} paginas")

print("Partiendo en chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"Generados {len(chunks)} chunks")

print("Creando embeddings (puede tomar 1-2 minutos la primera vez)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Construyendo base vectorial...")
db = Chroma.from_documents(chunks, embeddings)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

PROMPT_TEMPLATE = """Eres un asistente para serenos de la municipalidad de Chorrillos.
Responde unicamente con base en el siguiente contexto extraido de documentos oficiales.
Si la respuesta no esta en el contexto, di claramente: "No tengo esa informacion en los documentos disponibles."
No inventes informacion. Cita el numero de pagina cuando sea posible.

Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:"""

print("\nListo. Hazme preguntas (escribe salir para terminar).\n")

while True:
    pregunta = input("Pregunta: ")
    if pregunta.lower() in ["salir", "exit", "quit"]:
        break

    # 1. Buscar los 4 chunks mas parecidos a la pregunta
    relevantes = db.similarity_search(pregunta, k=4)

    # 2. Armar el contexto con los chunks encontrados
    bloques = []
    for d in relevantes:
        pagina = d.metadata.get("page", "?")
        bloques.append(f"[Pagina {pagina}] {d.page_content}")
    contexto = "\n\n".join(bloques)

    # 3. Llamar a Gemini con el prompt completo
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pregunta=pregunta)
    respuesta = llm.invoke(prompt)

    # 4. Mostrar respuesta y fuentes
    print("\nRespuesta:", respuesta.content)
    paginas = sorted(set([d.metadata.get("page", "?") for d in relevantes]))
    fuentes = sorted(set([os.path.basename(d.metadata.get("source", "?")) for d in relevantes]))
    print(f"Fuentes: {fuentes} | Paginas: {paginas}\n")