import streamlit as st
import os
import requests # Importe requests para o download dinâmico (se for usar)
from openai import OpenAI
# Removida a importação de 'google.colab.userdata' e 'drive', pois não existem no Streamlit Cloud.

# =========================================================================
# 1. FUNÇÕES DE SUPORTE E CONSTANTES (RAG)
# =========================================================================

# --- CONSTANTES ---
SYSTEM_PROMPT_FINAL = (
    "Você é um **Analista de Documentos Filosóficos** altamente rigoroso e imparcial. "
    "Sua única fonte de informação é a TRANSCRICÃO COMPLETA fornecida. "
    "Sua missão é extrair, sintetizar e articular as respostas de forma completa e profissional. "
    # ... (Resto do seu SYSTEM_PROMPT_FINAL)
    "**REGRAS DE ANÁLISE DO CONTEXTO:**"
    "1. **Síntese Obrigatória**: O documento é uma transcrição de fala informal e esparsa. Você DEVE sintetizar o conceito, rastreando o significado em diferentes partes do texto."
    "2. **Fidelidade Total**: Responda APENAS com base na informação fornecida na transcrição. Não use conhecimento externo sobre filosofia, psicologia ou temas esotéricos."
    "3. **Uso de Exemplos**: Se a pergunta pedir exemplos dos participantes (ex: SPEAKER_04, SPEAKER_08), você DEVE encontrá-los na transcrição e incorporá-los."
    "4. **Tratamento de Informação Faltante**: Se a resposta for insuficiente ou não estiver clara no texto, sua resposta OBRIGATÓRIA é: 'A informação solicitada não foi encontrada explicitamente no arquivo.' "
    "**FORMATO DE SAÍDA**: Entregue uma resposta formatada e clara, utilizando listas ou parágrafos concisos."
)

# --- FUNÇÃO RAG ---
def answer_rag_question(query: str, context_text: str, client: OpenAI) -> str:
    """Função para gerar a resposta RAG usando o cliente OpenAI fornecido pelo usuário."""
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

# --- FUNÇÃO DE CARREGAMENTO DE CONTEXTO ---
@st.cache_data
def load_context_from_txt(path: str) -> str:
    """Lê o conteúdo completo do arquivo TXT do repositório."""
    try:
        # Se você estiver usando o download dinâmico do Drive, substitua esta função
        # pela função 'load_context_from_url' (que usa requests)
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"ERRO ao carregar o conteúdo do TXT: {e}"


# =========================================================================
# 2. INTERFACE PRINCIPAL (Chat)
# =========================================================================

def run_chat_interface(client, CONTEXT_TEXT):
    """Contém toda a lógica de exibição e interação do chat."""

    st.title("Chatbot RAG - Análise de Documentos Filosóficos")
    st.caption(f"Contexto carregado: {len(CONTEXT_TEXT)} caracteres.")

    # Resto da lógica de chat (histórico, loop de mensagens, chamada RAG)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            {"role": "assistant", "content": "Olá! Sua chave foi validada. Faça uma pergunta sobre o documento."}
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Lógica de input e chamada RAG
    if prompt := st.chat_input("Digite sua pergunta aqui...", key="chat_input_principal"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando o contexto..."):
                response = answer_rag_question(prompt, CONTEXT_TEXT, client)

            st.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})

# =========================================================================
# 3. LÓGICA DE EXECUÇÃO E AUTENTICAÇÃO (O CORPO PRINCIPAL DO APP)
# =========================================================================

# Configurações de Path (Usando o arquivo local do repositório)
FILE_PATH = "palestras.txt"
CONTEXT_TEXT = load_context_from_txt(FILE_PATH)

if CONTEXT_TEXT.startswith("ERRO"):
    st.error("Erro ao carregar o contexto. Verifique se 'palestras.txt' está no repositório.")
    st.stop()


# 2. Lógica de Autenticação (Chave de API)
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

if not st.session_state["openai_api_key"]:
    
    st.title("Bem-vindo ao Analista de Documentos")
    st.warning("Para iniciar, por favor, insira sua chave secreta da OpenAI (ex: sk-...).")

    with st.form("api_key_form"):
        api_key_input = st.text_input(
            "Chave de API da OpenAI", 
            type="password", 
            key="api_key_temp"
        )
        submit_button = st.form_submit_button("Iniciar Chat")

        if submit_button and api_key_input:
            try:
                # Tenta inicializar o cliente para validar a chave
                client_temp = OpenAI(api_key=api_key_input.strip())
                client_temp.models.list() # Uma chamada simples para validar

                st.session_state["openai_api_key"] = api_key_input.strip()
                st.success("Chave validada! Reiniciando o app...")
                st.experimental_rerun()

            except Exception as e:
                st.error("Chave inválida. Verifique sua chave e tente novamente.")

else:
    # 3. Se a chave estiver na sessão, inicializa o cliente e roda a interface
    user_client = OpenAI(api_key=st.session_state["openai_api_key"])
    
    # Botão de Sair na Sidebar
    if st.sidebar.button("Trocar Chave / Sair"):
        st.session_state["openai_api_key"] = ""
        st.session_state.messages = []
        st.experimental_rerun()

    # Roda a interface principal do chat
    run_chat_interface(user_client, CONTEXT_TEXT)
         # Roda a interface principal do chat
         # TUDO O QUE FOR INTERAÇÃO DO CHAT DEVE OCORRER DENTRO DESTA FUNÇÃO
    run_chat_interface(user_client, CONTEXT_TEXT) 
