import streamlit as st
import os
import io
import re
import pdfplumber
from openai import OpenAI
# Removida a importação de google.colab e requests, que são desnecessárias para o deploy.

# =========================================================================
# 1. FUNÇÕES DE SUPORTE E CONSTANTES (RAG)
# =========================================================================

SEPARADOR = "\n\n--- FIM DO DOCUMENTO ---\n\n"

# --- FUNÇÕES DE LIMPEZA E EXTRAÇÃO (Adaptadas para Streamlit) ---

def limpar_transcricao(texto_bruto):
    """Remove conteúdo entre colchetes, ruído de fala e pontuação."""
    # Padrão Regex para remover [TEXTO]:
    PADRAO_COLCHETES = r'\[.*?\]:'
    texto_limpo = re.sub(PADRAO_COLCHETES, '', texto_bruto)
    
    PALAVRAS_DE_RUIDO = [' né', ' Né?', ' ok', ' Ok', ' Ok?', ' E aí', ' e aí', ' Oi', ' Tchau', ' obrigado', ' Obrigado', ' ...', ' pô', ' ah', ' Ah', ' Amém', ' .', ' ,']

    for palavra in PALAVRAS_DE_RUIDO:
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
        # O Streamlit mostrará este erro no log
        print(f"ERRO PDF: Falha na extração de {uploaded_file.name}: {e}")
        return None


@st.cache_data(show_spinner="Compilando e limpando documentos...")
def compile_uploaded_files(uploaded_files: list) -> str:
    """
    Processa uma lista de arquivos carregados (UploadedFile), aplica a limpeza e concatena.
    """
    if not uploaded_files:
        return ""

    compiled_text = ""
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        conteudo_bruto = None 

        if file_name.lower().endswith(".pdf"):
            conteudo_bruto = extrair_texto_pdf_streamlit(uploaded_file)
        
        elif file_name.lower().endswith(".txt"):
            try:
                # O read() do UploadedFile deve ser feito apenas uma vez por rerun
                conteudo_bruto = uploaded_file.read().decode("utf-8")
            except Exception as e:
                print(f"ERRO TXT: Falha ao ler o arquivo {file_name}: {e}")
                conteudo_bruto = None

        if conteudo_bruto:
            conteudo_limpo = limpar_transcricao(conteudo_bruto)
            
            if compiled_text:
                 compiled_text += SEPARADOR
            
            compiled_text += f"### Conteúdo da Fonte: {file_name} ###\n\n"
            compiled_text += conteudo_limpo

    return compiled_text


# --- CONSTANTES ---
SYSTEM_PROMPT_FINAL = ("Você é um **Analista de Documentos Filosóficos** altamente rigoroso e imparcial. "
"Sua única fonte de informação é a TRANSCRICÃO COMPLETA fornecida. "
"Sua missão é extrair, sintetizar e articular as respostas de forma completa e profissional. "
"**REGRAS DE ANÁLISE DO CONTEXTO:**"
"1. **Síntese Obrigatória**: O documento é uma transcrição de fala informal e esparsa. Você DEVE sintetizar o conceito, rastreando o significado em diferentes partes do texto."
"2. **Fidelidade Total**: Responda APENAS com base na informação fornecida na transcrição. Não use conhecimento externo sobre filosofia, psicologia ou temas esotéricos."
"3. **Uso de Exemplos**: Se a pergunta pedir exemplos dos participantes (ex: SPEAKER_04, SPEAKER_08), você DEVE encontrá-los na transcrição e incorporá-los."
"4. **Tratamento de Informação Faltante**: Se a resposta for insuficiente ou não estiver clara no texto, sua resposta OBRIGATÓRIA é: "
"'A informação solicitada não foi encontrada explicitamente no arquivo.' "
"**FORMATO DE SAÍDA**: Entregue uma resposta formatada e clara, utilizando listas ou parágrafos concisos."                     
)
# --- FUNÇÃO RAG (Inalterada) ---
def answer_rag_question(query: str, context_text: str, client: OpenAI) -> str:"""Função para gerar a resposta RAG usando o cliente OpenAI fornecido pelo usuário."""
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


# =========================================================================
# 2. INTERFACE PRINCIPAL (Chat)
# =========================================================================

def run_chat_interface(client, CONTEXT_TEXT):
    """Contém toda a lógica de exibição e interação do chat, usando o contexto dinâmico."""

    st.title("Assistente inteligente para se aprofundar na Heulosofia")
    st.caption(f"Contexto carregado: {len(CONTEXT_TEXT)} caracteres.")
    
    # Verifica se o contexto é apenas uma mensagem de aviso/erro
    is_context_ready = not CONTEXT_TEXT.startswith("Nenhum documento") and len(CONTEXT_TEXT) > 100

    if not is_context_ready:
        st.warning("Carregue arquivos válidos na barra lateral para ativar o chat. O assistente só pode responder com base no conteúdo carregado.")
    

    # Resto da lógica de chat (histórico, loop de mensagens, chamada RAG)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            {"role": "assistant", "content": "Olá! Sua chave foi validada. Faça uma pergunta sobre a Heulosofia."}
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Lógica de input e chamada RAG
    if prompt := st.chat_input("Digite sua pergunta aqui...", key="chat_input_principal", disabled=not is_context_ready):
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

# REMOVIDA A LÓGICA DE CARREGAMENTO DE ARQUIVO ESTÁTICO AQUI!

# 2. Lógica de Autenticação (Chave de API)
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

if not st.session_state["openai_api_key"]:
    
    st.title("Bem-vindo ao sua assistente da Heulosofia")
    st.warning("Para iniciar, por favor, insira sua chave API da OpenAI.")

    with st.form("api_key_form"):
        api_key_input = st.text_input(
            "Chave de API da OpenAI", 
            type="password", 
            key="api_key_temp"
        )
        submit_button = st.form_submit_button("Iniciar Chat")

        if submit_button and api_key_input:
            try:
                client_temp = OpenAI(api_key=api_key_input.strip())
                client_temp.models.list() 

                st.session_state["openai_api_key"] = api_key_input.strip()
                st.success("Chave validada! Reiniciando o app...")
                st.rerun() # CORRIGIDO: de experimental_rerun para rerun

            except Exception as e:
                st.error("Chave inválida. Verifique sua chave e tente novamente.")

else:
    # --- FLUXO PRINCIPAL: CLIENTE AUTENTICADO ---
    
    user_client = OpenAI(api_key=st.session_state["openai_api_key"])
    
    # -------------------------------------------------------------
    # NOVO: Upload e Geração de Contexto na Sidebar
    # -------------------------------------------------------------
    
    with st.sidebar:
        st.subheader("🛠️ Contexto de Documentos")
        
        uploaded_files = st.file_uploader(
            "Arraste e solte seus PDFs/TXTs para análise:",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="file_uploader_context"
        )
        
        # Botão de Sair na Sidebar
        if st.button("Trocar Chave / Sair"):
            st.session_state["openai_api_key"] = ""
            st.session_state.messages = []
            st.rerun() # CORRIGIDO: de experimental_rerun para rerun

    # Gera o CONTEXTO DINÂMICO
    if uploaded_files:
        # Chama a função de compilação
        current_context = compile_uploaded_files(uploaded_files)
        
        if not current_context or len(current_context) < 100:
            st.warning("Arquivos carregados, mas nenhum texto válido foi extraído. Por favor, verifique o formato dos documentos.")
            current_context = "ERRO: Nenhum contexto válido disponível."
        else:
             st.success(f"Contexto RAG pronto! Total de caracteres: {len(current_context)}")

    else:
        # Contexto de Aviso (quando nenhum arquivo foi carregado)
        current_context = "Nenhum documento carregado. O assistente não tem conteúdo para análise."
        st.info("Carregue seus documentos na barra lateral para iniciar a análise RAG.")
    
    # -------------------------------------------------------------
    
    # Roda a interface principal do chat
    run_chat_interface(user_client, current_context)

    # Roda a interface principal do chat, passando o contexto DINÂMICO
    run_chat_interface(user_client, current_context)
