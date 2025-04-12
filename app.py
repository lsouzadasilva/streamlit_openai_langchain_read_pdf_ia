from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os


def config_tela():
    st.set_page_config(
        page_title='J.A.R.V.I.S RAG PDF',
        page_icon='🤖'
    )
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Assistente de leitura PDF 🤖</h1>", unsafe_allow_html=True)
    st.divider()
config_tela()


def ocult_menu():
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
ocult_menu()


st.sidebar.markdown("<h2 style='color: #A67C52;'>J.A.R.V.I.S 🤖</h2>", unsafe_allow_html=True)


tab_home, tab_config = st.sidebar.tabs(["Home", "Configurações"])

with tab_home:
    def sobre():
        st.markdown("<h2 style='color: #4B8BBE;'>Sobre o projeto</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size: 0.9em; line-height: 1.4;'>
        <b>Oraclo PDF</b> é um assistente inteligente que:<br>
        • Analisa seus documentos PDF<br>
        • Responde perguntas precisas<br>
        • Mostra as páginas de referência<br><br>
        Tecnologias:<br>
        <span style='color: #4B8BBE;'>• OpenAI • LangChain • FAISS • Streamlit</span>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        st.sidebar.markdown("""
        **Desenvolvido por Leandro Souza**  
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/leandro-souza-bi/)
        [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lsouzadasilva/streamlit_openai_langchain_read_pdf)
        """)
    sobre()

      
with tab_config: 
    def api_open_ai():
        if 'api_key' not in st.session_state:
            st.session_state.api_key = None

        api_key = st.text_input("API Key", type="password")
        if api_key:
            st.session_state.api_key = api_key
            st.success('Chave salva com sucesso!')
    api_open_ai()
      
    def carregar_pdf():
        uploaded_file = st.file_uploader("Carregar PDF", type=["pdf"])
        accept_multiple_files=True
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
                
            os.unlink(tmp_file_path)
            return splits
        return None

    documents = carregar_pdf()
    

def setup_retriever(documents):
    if documents and st.session_state.api_key:
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
        )
        retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 25})
        return retriever
    return None

retriever = setup_retriever(documents)

def join_documents(docs_dict):
    if 'contexto' in docs_dict:
        joined_content = []
        for doc in docs_dict['contexto']:
            page_num = doc.metadata.get('page', 'N/A')
            content = f"Página {page_num}:\n{doc.page_content}"
            joined_content.append(content)
        docs_dict['contexto'] = "\n\n".join(joined_content)
    return docs_dict

def create_prompt_template():
    return ChatPromptTemplate.from_template(
        '''Responda as perguntas se baseando no contexto fornecido. 
        Sempre inclua o número da página de onde a informação foi extraída.
        
        Contexto: {contexto}
        
        Pergunta: {pergunta}
        
        Formate sua resposta incluindo entre parênteses (página X) 
        a referência da página sempre que possível.'''
    )

prompt_template = create_prompt_template()

def main_chat():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Faça sua pergunta sobre o PDF"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if retriever and st.session_state.api_key:
            try:
                setup = RunnableParallel({
                    "contexto": retriever,
                    "pergunta": RunnablePassthrough()
                })
                
                chain = (
                    setup
                    | RunnableLambda(join_documents)
                    | prompt_template
                    | ChatOpenAI(openai_api_key=st.session_state.api_key, temperature=0.3)
                )
                
                with st.chat_message("assistant"):
                    response = chain.invoke(user_input)
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
            
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"Ocorreu um erro: {str(e)}")
        else:
            with st.chat_message("assistant"):
                st.error("Por favor, carregue um PDF e insira uma API Key válida.")

main_chat()
