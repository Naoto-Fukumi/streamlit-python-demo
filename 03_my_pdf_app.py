import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.llms import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback

QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection"


# ローカルへの保存
client = QdrantClient(path="./local_qdrant")

# qdrant cloud への保存 (次の章で詳しく話します)
client = QdrantClient(
    url="https://3c447ffa-44b7-4f08-85d7-bd61f13d05f4.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="Dr9FsZx21w4niJg2OH3WpoOtRzD2EiIzTeFRhvsW_9DJ_1GYmQhbUg"
)

def init_page():
    st.set_page_config(
        page_title="Youtube Summarizer",
        page_icon="🤗"
    )
    st.header("Youtube Summarizer 🤗")
    st.sidebar.title("Options")
    st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo"
    else:
        st.session_state.model_name = "gpt-4"

    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)

def sidebar():
    selection = st.sidebar.radio('Go to', ['PDF Upload', 'Ask My PDF'])
    if selection == 'PDF Upload':
        page_pdf_upload_and_build_vector_db()
    elif selection == 'Ask My PDF':
        page_ask_my_pdf()


def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)

    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        # コレクションが存在しない場合、新しく作成します
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')
    
    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME, 
        embeddings=OpenAIEmbeddings()
    )

def build_qa_model(llm):
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        # "mmr",  "similarity_score_threshold" などもある
        search_type='similarity',
        # 文書を何個取得するか (default: 4)
        search_kwargs={'k':10}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )


def ask(qa, query):
    with get_openai_callback() as cb:
        # query / result / source_documents
        answer = qa(query)

    return answer, cb.total_cost

def build_vector_store(pdf_text):
    qdrant = load_qdrant()
    qdrant.add_texts(pdf_text)
    # query = "トヨタとの決算について"
    # docs = qdrant.similarity_search(query=query, k=2)
    # for i in docs:
    #     print({"content": i.page_content, "metadata": i.metadata})


def page_pdf_upload_and_build_vector_db():
    # ここにPDFアップロードページの実装をする
    # print('test')
    st.title("PDF Upload")
    container = st.container()
    with container:
        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("Loading PDF ..."):
                build_vector_store(pdf_text)

def page_ask_my_pdf():
    # ここにChatGPTに質問を投げるページの実装をする
    st.title("Ask My PDF(s)")

    llm = select_model()
    container = st.container()
    response_container = st.container()

    with container:
        query = st.text_input("Query: ", key="input")
        if not query:
            answer = None
        else:
            qa = build_qa_model(llm)
            if qa:
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = ask(qa, query)
                st.session_state.costs.append(cost)
            else:
                answer = None

        if answer:
            with response_container:
                st.markdown("## Answer")
                st.write(answer)


def get_pdf_text():
    upload_file = st.file_uploader(
        label='PDFをアップロードしてください😇',
        type='pdf' # アップロードを許可する拡張子 (複数設定可)
    )
    if upload_file:
        pdf_reader = PdfReader(upload_file)
        # print(pdf_reader.pages)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=st.session_state.model_name,
            chunk_size=250,
            chunk_overlap=0,
        )
        return text_splitter.split_text(text)
    else:
        return None



def main():
    init_page()

    sidebar()

    # llm = select_model()

    # container = st.container()

if __name__ == '__main__':
    main()