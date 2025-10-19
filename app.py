import streamlit as st
import os
from openai import OpenAI
# Removida a importação de 'google.colab.userdata' e 'drive', pois não existem no Streamlit Cloud.

# --- 1. CONFIGURAÇÃO DE CONSTANTES ---

# No Streamlit Cloud, o arquivo deve estar no mesmo diretório ou ser baixado.
# Assumimos que o arquivo 'palestras.txt' será adicionado manualmente ao repositório GitHub.
# Se o arquivo for MUITO grande (acima de 100MB), a Estratégia B (gdown/API) abaixo será necessária.
FILE_PATH = "palestras.txt"

SYSTEM_PROMPT_FINAL = (
    "Você é um **Analista de Documentos Filosóficos** altamente rigoroso e imparcial. "
    "Sua única fonte de informação é a TRANSCRICÃO COMPLETA fornecida. "
    # ... (Resto do seu SYSTEM_PROMPT_FINAL)
    "Sua missão é extrair, sintetizar e articular as respostas de forma completa e profissional. "
    "**REGRAS DE ANÁLISE DO CONTEXTO:**"
    "1. **Síntese Obrigatória**: O documento é uma transcrição de fala informal e esparsa. Você DEVE sintetizar o conceito, rastreando o significado em diferentes partes do texto."
    "2. **Fidelidade Total**: Responda APENAS com base na informação fornecida na transcrição. Não use conhecimento externo sobre filosofia, psicologia ou temas esotéricos."
    "3. **Uso de Exemplos**: Se a pergunta pedir exemplos dos participantes (ex: SPEAKER_04, SPEAKER_08), você DEVE encontrá-los na transcrição e incorporá-los."
    "4. **Tratamento de Informação Faltante**: Se a resposta for insuficiente ou não estiver clara no texto, sua resposta OBRIGATÓRIA é: "
    "'A informação solicitada não foi encontrada explicitamente no arquivo.' "
    "**FORMATO DE SAÍDA**: Entregue uma resposta formatada e clara, utilizando listas ou parágrafos concisos."
)

# --- 2. FUNÇÕES DE CARREGAMENTO E RAG ---

@st.cache_data
def load_context_from_txt(path: str) -> str:
    """Lê o conteúdo completo do arquivo TXT e usa cache do Streamlit."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"ERRO CRÍTICO: Não foi possível carregar o arquivo de contexto TXT: {e}")
        return f"ERRO ao carregar o conteúdo do TXT: {e}"

@st.cache_resource
def get_openai_client():
    """Inicializa o cliente OpenAI usando a API Key dos Secrets do Streamlit Cloud."""
    try:
        # A chave 'OPENAI_API_KEY' é carregada automaticamente pelo Streamlit
        api_key = os.environ.get("OPENAI_API_KEY") 
        if not api_key:
            st.error("Chave 'OPENAI_API_KEY' não configurada nos Secrets do Streamlit Cloud.")
            return None
        return OpenAI(api_key=api_key.strip())
    except Exception as e:
        st.error(f"Erro ao inicializar o Cliente OpenAI: {e}")
        return None

def answer_rag_question(query: str, context_text: str, client: OpenAI) -> str:
    """Função para gerar a resposta RAG."""
    context_slice = context_text[:128000] # Limita para segurança de tokens

    full_user_message = (
        f"CONTEXTO COMPLETO DO DOCUMENTO (TRANSCRICÃO):\n---\n{context_slice}\n---\n\n"
        f"PERGUNTA A SER ANALISADA E RESPONDIDA: {query}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FINAL},
        {"role": "user", "content": full_user_message}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERRO na chamada à API da OpenAI: {e}"

# --- 3. LÓGICA PRINCIPAL DO STREAMLIT (Interface de Chat) ---

client = get_openai_client()
CONTEXT_TEXT = load_context_from_txt(FILE_PATH)

st.title("Chatbot RAG - Análise de Documentos Filosóficos")
st.caption(f"Contexto carregado: {len(CONTEXT_TEXT)} caracteres.")

if client is None or CONTEXT_TEXT.startswith("ERRO"):
    st.warning("Verifique a chave OpenAI e a presença do arquivo de contexto no repositório.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "assistant", "content": "Olá! Sou o Analista. Faça uma pergunta sobre o documento."}
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua pergunta aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analisando o contexto..."):
            response = answer_rag_question(prompt, CONTEXT_TEXT, client)

        st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
