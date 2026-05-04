import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

CARPETA_DOCS = "documentos"
CARPETA_DB = "chroma_db"

# --- Configuracion de la pagina ---
st.set_page_config(
    page_title="Asistente Serenazgo Chorrillos",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Asistente Serenazgo Chorrillos")
st.caption("Consulta basada en documentos oficiales de la Municipalidad de Chorrillos")

# --- Carga de modelo y base vectorial (solo una vez) ---
@st.cache_resource
def cargar_sistema():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        encode_kwargs={"normalize_embeddings": True}
    )

    if Path(CARPETA_DB).exists() and any(Path(CARPETA_DB).iterdir()):
        db = Chroma(persist_directory=CARPETA_DB, embedding_function=embeddings)
    else:
        loader = PyPDFDirectoryLoader(f"{CARPETA_DOCS}/")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        db = Chroma.from_documents(chunks, embeddings, persist_directory=CARPETA_DB)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )
    return db, llm

with st.spinner("Cargando sistema..."):
    db, llm = cargar_sistema()

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

# --- Historial de conversacion ---
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar mensajes anteriores
for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])
        if msg.get("fuentes"):
            st.caption(f"📄 Fuentes: {msg['fuentes']}")

# --- Input del usuario ---
pregunta = st.chat_input("Escribe tu pregunta aquí...")

if pregunta:
    # Mostrar pregunta del usuario
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando en documentos..."):
            relevantes = db.similarity_search(pregunta, k=6)

            bloques = []
            for d in relevantes:
                pagina = d.metadata.get("page", "?")
                bloques.append(f"[Pagina {pagina}] {d.page_content}")
            contexto = "\n\n".join(bloques)

            prompt = PROMPT_TEMPLATE.format(contexto=contexto, pregunta=pregunta)
            respuesta = llm.invoke(prompt)

            paginas = sorted(set([d.metadata.get("page", "?") for d in relevantes]))
            fuentes = sorted(set([os.path.basename(d.metadata.get("source", "?")) for d in relevantes]))
            fuentes_str = f"{fuentes} | Páginas: {paginas}"

        st.markdown(respuesta.content)
        st.caption(f"📄 Fuentes: {fuentes_str}")

    st.session_state.mensajes.append({
        "rol": "assistant",
        "contenido": respuesta.content,
        "fuentes": fuentes_str
    })

# --- Sidebar con info del proyecto ---
with st.sidebar:
    st.header("ℹ️ Sobre este asistente")
    st.write(
        "Este asistente responde preguntas sobre seguridad ciudadana en Chorrillos "
        "usando documentación oficial del distrito."
    )
    st.subheader("Documentos cargados")
    st.write("- Reglamento de Organización y Funciones (ROF)")
    st.write("- Plan de Acción Distrital de Seguridad Ciudadana 2024-2027")
    st.subheader("Ejemplos de preguntas")
    st.write("- ¿Cuántas comisarías hay en Chorrillos?")
    st.write("- ¿Cuáles son los principales delitos del distrito?")
    st.write("- ¿Qué hace la Subgerencia de Serenazgo?")
    st.write("- ¿Qué es el CODISEC?")

    if st.button("🗑️ Limpiar conversación"):
        st.session_state.mensajes = []
        st.rerun()