from flask import Flask, request, jsonify
import os
import hashlib
import nltk
from langdetect import detect as lang_detect

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = Flask(__name__)
nltk.download('punkt')

GOOGLE_API_KEY = "AIzaSyDeWL_xzEevDYZMIJLQEwJxT6eP8wuiSew"  # Your API key
chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-flash")
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)

def get_db_hash_from_url(url):
    return hashlib.md5(url.encode()).hexdigest()

def get_db_hash_from_urls(urls):
    combined = ''.join(sorted(urls))
    return hashlib.md5(combined.encode()).hexdigest()

def detect_transcript_language(docs):
    try:
        sample_text = docs[0].page_content
        lang_code = lang_detect(sample_text)
        print(f"Detected transcript language: {lang_code}")
        return lang_code
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "unknown"

def process_single_video(video_url):
    db_hash = get_db_hash_from_url(video_url)
    db_dir = f"./chroma_db_{db_hash}"

    if not os.path.exists(db_dir):
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=False,
            language=["bn", "en", "hi", "ta", "ur", "te"]
        )
        docs = loader.load()
        if not docs:
            raise Exception(f"No transcript found for video: {video_url}")

        lang_code = detect_transcript_language(docs)
        for doc in docs:
            doc.metadata["source_video"] = video_url
            doc.metadata["language"] = lang_code

        chunks = text_splitter.split_documents(docs)
        db = Chroma.from_documents(chunks, embedding_model, persist_directory=db_dir)
        db.persist()

    return Chroma(persist_directory=db_dir, embedding_function=embedding_model)

def process_multiple_videos(video_urls):
    all_docs = []
    db_hash = get_db_hash_from_urls(video_urls)
    db_dir = f"./chroma_db_multi_{db_hash}"

    if not os.path.exists(db_dir):
        for video_url in video_urls:
            try:
                loader = YoutubeLoader.from_youtube_url(
                    video_url,
                    add_video_info=False,
                    language=["bn", "en", "hi", "ta", "ur", "te"]
                )
                docs = loader.load()
                if not docs:
                    raise Exception(f"No transcript found for video: {video_url}")

                lang_code = detect_transcript_language(docs)
                for doc in docs:
                    doc.metadata["source_video"] = video_url
                    doc.metadata["language"] = lang_code

                all_docs.extend(docs)
            except Exception as e:
                print(f"Error processing video {video_url}: {e}")
                continue

        if not all_docs:
            raise Exception("No transcripts found for any of the provided videos.")

        chunks = text_splitter.split_documents(all_docs)
        db = Chroma.from_documents(chunks, embedding_model, persist_directory=db_dir)
        db.persist()

    return Chroma(persist_directory=db_dir, embedding_function=embedding_model)

def create_rag_chain_with_timestamps(db_connection):
    retriever = db_connection.as_retriever(search_kwargs={"k": 5})

    def format_docs_with_timestamps(docs):
        formatted = []
        for doc in docs:
            ts = doc.metadata.get("start") or doc.metadata.get("timestamp") or "00:00:00"
            if isinstance(ts, (int, float)):
                h = int(ts // 3600)
                m = int((ts % 3600) // 60)
                s = int(ts % 60)
                ts_str = f"{h:02d}:{m:02d}:{s:02d}"
            else:
                ts_str = str(ts)
            formatted.append(f"[{ts_str}] {doc.page_content}")
        return "\n\n".join(formatted)

    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""আপনি একজন সহায়ক সহকারী। প্রদত্ত প্রসঙ্গ এবং টাইমস্ট্যাম্প ব্যবহার করে প্রশ্নের উত্তর দিন। যদি প্রসঙ্গে [00:01:23] এর মতো টাইমস্ট্যাম্প থাকে, তবে তা উল্লেখ করুন।"""),
        HumanMessagePromptTemplate.from_template("""প্রসঙ্গ: {context}

প্রশ্ন: {question}

উত্তর:""")
    ])

    output_parser = StrOutputParser()

    return (
        {"context": retriever | format_docs_with_timestamps, "question": RunnablePassthrough()}
        | chat_template
        | chat_model
        | output_parser
    )

def create_rag_chain_simple(db_connection):
    retriever = db_connection.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""আপনি একজন সহায়ক সহকারী। প্রদত্ত প্রসঙ্গ ব্যবহার করে প্রশ্নের উত্তর দিন।"""),
        HumanMessagePromptTemplate.from_template("""প্রসঙ্গ: {context}

প্রশ্ন: {question}

উত্তর:""")
    ])

    output_parser = StrOutputParser()

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | chat_template
        | chat_model
        | output_parser
    )

@app.route('/ask_individual', methods=['POST'])
def ask_individual():
    data = request.json
    video_url = data.get('video_url')
    question = data.get('question', '')

    if not video_url or not question:
        return jsonify({"error": "ভিডিও URL এবং প্রশ্ন প্রয়োজন।"}), 400

    try:
        db = process_single_video(video_url)
        rag_chain = create_rag_chain_with_timestamps(db)
        answer = rag_chain.invoke(question)
        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask_multiple', methods=['POST'])
def ask_multiple():
    data = request.json
    video_urls = data.get('video_urls', [])
    question = data.get('question', '')

    if not video_urls or len(video_urls) == 0 or not question:
        return jsonify({"error": "ভিডিও URL গুলির তালিকা এবং প্রশ্ন প্রয়োজন।"}), 400

    try:
        db = process_multiple_videos(video_urls)
        rag_chain = create_rag_chain_simple(db)
        answer = rag_chain.invoke(question)
        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
