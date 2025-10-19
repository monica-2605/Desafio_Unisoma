import streamlit as st
import os
import io
import re
import pdfplumber
from openai import OpenAI
# Removida a importação de google.colab e requests

# =========================================================================
# A. FUNÇÕES DE PRÉ-PROCESSAMENTO (MIGRADAS DO SEU CÓDIGO COLAB)
# =========================================================================

SEPARADOR = "\n\n--- FIM DO DOCUMENTO ---\n\n"

def limpar_transcricao(texto_bruto):
    """Remove conteúdo entre colchetes, ruido de fala e pontuação."""
    PADRAO_COLCHETES = r'\[.*?\]:'
    texto_limpo = re.sub(PADRAO_COLCHETES, '', texto_bruto)
    PALAVRAS_DE_RUIDO = [' né', ' Né?', ' ok', ' Ok', ' Ok?', ' E aí', ' e aí', ' Oi', ' Tchau', ' obrigado', ' Obrigado', ' ...', ' pô', ' ah', ' Ah', ' Amém', ' .', ' ,']

    for palavra in PALAVRAS_DE_RUIDO:
        # Substitui a palavra por um espaço simples
        texto_limpo = texto_limpo.replace(palavra, ' ')

    return texto_limpo

def extrair_texto_pdf_streamlit(uploaded_file):
    """Extrai texto de um UploadedFile PDF, lendo o conteúdo em memória."""
    texto_extraido = ""
    try:
        # Usa io.BytesIO para ler o arquivo binário do Streamlit na memória
        pdf_bytes = io.BytesIO(uploaded_file.read())
        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                texto_extraido += page.extract_text() + "\n"
        return texto_extraido
    except Exception as e:
        # st.error não funciona dentro desta função @st.cache_data, mas é bom para debug.
        print(f"ERRO PDF: Falha na extração de {uploaded_file.name}: {e}")
        return None


@st.cache_data(show_spinner="Compilando e limpando documentos...")
def compile_uploaded_files(uploaded_files: list) -> str:
    """
    Processa uma lista de arquivos carregados (UploadedFile) e concatena o conteúdo limpo.
    """
    if not uploaded_files:
        return ""

    compiled_text = ""
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        conteudo_bruto = None 

        # 1. Extração do Conteúdo Bruto (PDF ou TXT)
        if file_name.lower().endswith(".pdf"):
            conteudo_bruto = extrair_texto_pdf_streamlit(uploaded_file)

        elif file_name.lower().endswith(".txt"):
            try:
                # Usa .read().decode() para ler o UploadedFile como texto
                conteudo_bruto = uploaded_file.read().decode("utf-8")
            except Exception as e:
                print(f"ERRO TXT: Falha ao ler o arquivo {file_name}: {e}")
                conteudo_bruto = None

        # 2. Limpeza e Concatenação
        if conteudo_bruto:
            conteudo_limpo = limpar_transcricao(conteudo_bruto)
            
            # Adiciona o separador antes de cada novo arquivo (exceto o primeiro)
            if compiled_text:
                 compiled_text += SEPARADOR
            
            # Adiciona um cabeçalho para identificação
            compiled_text += f"### Conteúdo da Fonte: {file_name} ###\n\n"
            compiled_text += conteudo_limpo

    return compiled_text

# --- FUNÇÃO OBSOLETA (Removida) ---
# Você não precisa mais de load_context_from_txt, pois usaremos a compilação em tempo real.
# Apenas a remova completamente do seu código final.


# =========================================================================
# B. RESTANTE DO CÓDIGO (ADAPTAÇÕES NECESSÁRIAS)
# =========================================================================

# --- FUNÇÃO RAG (Sem Alterações) ---
# ... (Mantenha a função answer_rag_question inalterada) ...


# --- INTERFACE PRINCIPAL (Chat) ---
def run_chat_interface(client, CONTEXT_TEXT):
    # ... (Mantenha esta função, mas ela usará o CONTEXT_TEXT dinâmico) ...
    pass


# --- LÓGICA DE EXECUÇÃO E AUTENTICAÇÃO (O CORPO PRINCIPAL DO APP) ---

# REMOVA O CONTEXTO ESTÁTICO!
# FILE_PATH = "palestras.txt" <-- REMOVA
# CONTEXT_TEXT = load_context_from_txt(FILE_PATH) <-- REMOVA

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

# ... (Lógica de Autenticação - MANTENHA INALTERADA) ...

else:
    user_client = OpenAI(api_key=st.session_state["openai_api_key"])
    
    # -------------------------------------------------------------
    # NOVO: Upload e Geração de Contexto
    # -------------------------------------------------------------
    
    with st.sidebar:
        st.subheader("🛠️ Contexto de Documentos")
        
        uploaded_files = st.file_uploader(
            "Arraste e solte seus PDFs/TXTs para análise:",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="file_uploader_context"
        )
    
    # Gera o CONTEXTO DINÂMICO
    if uploaded_files:
        current_context = compile_uploaded_files(uploaded_files)
        
        if not current_context:
            st.warning("Nenhum texto válido foi extraído dos documentos. Por favor, verifique os arquivos.")
            current_context = "ERRO: Nenhum contexto válido disponível."

    else:
        # Contexto de Aviso (quando nenhum arquivo foi carregado)
        current_context = "Nenhum documento carregado. Por favor, carregue os arquivos na barra lateral para análise."
        st.info("Carregue seus documentos na barra lateral para iniciar a análise RAG.")
    
    # -------------------------------------------------------------
    
    # Botão de Sair na Sidebar
    if st.sidebar.button("Trocar Chave / Sair"):
        st.session_state["openai_api_key"] = ""
        st.session_state.messages = []
        st.experimental_rerun()

    # Roda a interface principal do chat, passando o contexto DINÂMICO
    run_chat_interface(user_client, current_context)
