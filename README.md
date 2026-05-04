# Serenos Bot

Asistente IA para serenos municipales. Responde preguntas operativas (protocolos, ordenanzas, contactos de emergencia) usando RAG sobre documentación pública de la municipalidad.

**Estado:** En desarrollo. Piloto inicial planeado en Chorrillos, Lima.

## Stack
- Python 3.13
- Google Gemini (LLM)
- LangChain (orquestación) — próximamente
- ChromaDB (base vectorial) — próximamente
- Streamlit (interfaz web) — próximamente

## Motivación
Los serenos resuelven incidentes consultando protocolos, ordenanzas y directorios que están dispersos en PDFs y manuales. Un asistente que responda en segundos, citando la fuente exacta, ahorra tiempo en cada incidente y reduce errores en la operación diaria.

## Roadmap
- [x] Llamada básica al LLM
- [ ] System prompt anti-alucinación
- [ ] RAG con ordenanzas municipales
- [ ] Interfaz web (Streamlit)
- [ ] Piloto con usuarios reales
- [ ] Métricas de uso

## Autor
Marcelo Tolentino — [GitHub](https://github.com/Marcelo409)