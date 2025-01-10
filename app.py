import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import getpass

# Streamlit UI
st.title("RAG-Based Chatbot")
st.write("This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions based on the provided dataset.")

# Uploading the PDF File
uploaded_file = st.file_uploader("Upload a PDF file to load as the knowledge base", type=["pdf"])
if uploaded_file is not None:
    # Loading the data from the PDF
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("uploaded_file.pdf")
    documents = loader.load()
    st.success(f"Loaded {len(documents)} documents from the uploaded dataset.")

    # Setting up the OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = st.text_input("Enter your OpenAI API key:", type="password")
    
    # Checking if API key is set
    if os.environ.get("OPENAI_API_KEY"):
        st.info("API key set successfully.")

        # Creating the embeddings and set up FAISS vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()

        # Setting up the prompt template
        prompt = PromptTemplate(
            template="You are a helpful assistant. Based on the context below, answer the question concisely:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["context", "question"],
        )

        # Setting up the LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        # Setting up the retrieval QA pipeline
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        # Input query for chatbot
        query = st.text_input("Ask a question based on the uploaded dataset:")
        if st.button("Get Answer"):
            with st.spinner("Processing..."):
                result = qa({"query": query})
                answer = result['result']
                source_documents = result['source_documents']

                # Displaying the results
                st.subheader("Answer:")
                st.write(answer)

                st.subheader("Source Documents:")
                for doc in source_documents:
                    st.write(doc.page_content)
    else:
        st.error("Please provide a valid OpenAI API key.")
else:
    st.warning("Upload a PDF to proceed.")

#command to run the streamlit app- streamlit run app.py
