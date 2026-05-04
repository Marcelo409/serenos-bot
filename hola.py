import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hay una pelea en un parque, soy sereno de Chorrillos. ¿Qué hago?",
    config={"system_instruction": "Eres un asistente para serenos de la municipalidad de Chorrillos."}
)

print(response.text)
