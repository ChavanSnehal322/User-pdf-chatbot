import streamlit as st
import os
from dotenv import load_dotenv

import time
import json
import requests
from typing import List

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

# embeddings + index
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from Webpages import css, bot_template, user_template

# load_dotenv()
# hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# if not hf_token:
#     st.warning("Hugging face token not found ")

Embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

hf_model = "google/flan-t5-small"  # free

chunk_size = 500. # characters
chunk_overlap = 200
Top_k = 5

def load_embed_model(model_name:str):

    return SentenceTransformer(model_name)

def get_pdf_text(pdf_doc) -> str:
    text = ""
    for pdf in pdf_doc:
        try:
            pdf_reader = PdfReader(pdf)

            for page in pdf_reader.pages:
                ptext = page.extract_text()

                if ptext:
                    text += ptext + "\n"

        except Exception as e:
            st.error(f"Failed to read {getattr(f, 'name', 'file')}: {e} ")


    return text


def get_txt_chunks(text: str,  chunk_size, chunk_overlap) -> List[str]:
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    tokens = text.split("\n")
    chunks = []
    current = ""
    for line in tokens:
        if len(current) + len(line) + 1 <= chunk_size:
            current += (line + "\n")
        else:
            chunks.append(current.strip())
            # start new chunk with overlap
            overlap = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = overlap + line + "\n"
    if current.strip():
        chunks.append(current.strip())

    return chunks

@st.cache_resource
def load_gen_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    return tokenizer, model

def generate_answer(prompt: str, tokenizer, model, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def build_faiss_index(emb_model: SentenceTransformer, texts: List[str]):
    if len(texts) == 0:
        return None, None
    embeddings = emb_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product (cosine if we normalize)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, embeddings


def retrieve_similar(index, emb_model, embeddings_matrix, texts, query, top_k):
    if index is None:
        return []
    q_vec = emb_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(texts):
            continue
        results.append({"score": float(score), "text": texts[idx], "idx": int(idx)})
    return results


def hf_inference_generate(prompt: str, hf_model, max_new_tokens: int = 256):
    """
    Call Hugging Face Inference API using requests (avoids hf client incompatibilities).
    Returns generated text or raises an Exception on failure.
    """
    if not hf_token:
        raise RuntimeError("Hugging Face token missing. Set HUGGINGFACEHUB_API_TOKEN in .env.")
    
    url = f"https://router.huggingface.co/models/{hf_model}"

    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "return_full_text": False},
        "options": {"use_cache": False, "wait_for_model": True}
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HF Inference API error {resp.status_code}: {resp.text}")
    out = resp.json()
    # HF sometimes returns a list of dicts or a dict; normalize:
    if isinstance(out, dict) and "error" in out:
        raise RuntimeError(f"HF error: {out['error']}")
    # if list with dicts containing 'generated_text' or 'generated_text' directly
    if isinstance(out, list):
        # some endpoints return [{"generated_text": "..."}]
        if "generated_text" in out[0]:
            return out[0]["generated_text"]
        # or [{"generated_text": "...", ...}, ...] or strings (rare)
        # try best-effort:
        for item in out:
            if isinstance(item, dict) and "generated_text" in item:
                return item["generated_text"]
        # fallback to stringify
        return json.dumps(out)
    if isinstance(out, dict) and "generated_text" in out:
        return out["generated_text"]
    return str(out)


def Manage_userInput(User_ques):

    # handle the user question by initiating chat session 
    response = st.session_state.conversation({'question': User_ques})

    # return the response from the llm to the user and store the history in session
    #st.write(response)
    st.session_state.chat_history = response["chat_history"]

    for i, msgg in enumerate(st.session_state.chat_history):

        if i % 2 == 0:

             st.write(user_template.replace("{{MSG}}", "Hello AI" ), unsafe_allow_html=True)   
        else:
            
            st.write(bot_template.replace( "{{MSG}}", "Hello User"), unsafe_allow_html=True)

    

def main():

    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Session state for index and chat
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False
    if "texts" not in st.session_state:
        st.session_state.texts = []
    if "index" not in st.session_state:
        st.session_state.index = None
    if "emb_matrix" not in st.session_state:
        st.session_state.emb_matrix = None
    if "emb_model" not in st.session_state:
        st.session_state.emb_model = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {"role": "user"/"ai", "text": ...}

    st.header("Chat with multiple PDFs")
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_files = st.file_uploader("Upload your pdfs (multiple)", accept_multiple_files=True, type=["pdf"])
        
        model_choice = st.selectbox("Generation model (HF Inference)", [
            "google/flan-t5-small",
            "google/flan-t5-base",
            "EleutherAI/gpt-neo-125M"
        ])

        if st.button("Process"):
            if not pdf_files:
                st.sidebar.error("Upload at least one PDF first.")
            else:
                with st.spinner("Reading PDFs and building index (this may take a moment)..."):
                    text = get_pdf_text(pdf_files)
                    chunks = get_txt_chunks(text, chunk_size, chunk_overlap)
                    st.session_state.texts = chunks
                    # load embedding model (cached)
                    emb_model = load_embed_model(Embedding_model_name)
                    st.session_state.emb_model = emb_model
                    index, emb_matrix = build_faiss_index(emb_model, chunks)
                    st.session_state.index = index
                    st.session_state.emb_matrix = emb_matrix
                    st.session_state.index_ready = True
                st.sidebar.success(f"Processed {len(chunks)} chunks.")

    # Input area
    query = st.text_input("Ask a question about the uploaded PDFs:")

    if st.button("Ask") or (query and st.session_state.get("auto_ask_on_enter", True) and query):
        if not st.session_state.index_ready:
            st.error("Please upload PDFs and click Process first.")
        else:
            user_q = query.strip()
            if not user_q:
                st.warning("Please type a question.")
            else:
                st.session_state.chat_history.append({"role": "user", "text": user_q})
                
                # retrieve
                results = retrieve_similar(
                            st.session_state.index,
                            st.session_state.emb_model,
                            st.session_state.emb_matrix,
                            st.session_state.texts,
                            user_q,
                            top_k=Top_k
                )

                context = "\n\n. \n\n".join([f"(score:{r['score']:.3f})\n{r['text']}" for r in results])
                prompt = f"Use the following extracted passages from documents to answer the question. If the answer is not contained, say you don't know.\n\nPassages:\n{context}\n\nQuestion: {user_q}\n\nAnswer concisely:"
                try:
                    #answer = hf_inference_generate(prompt, hf_model=model_choice, max_new_tokens=300)
                    tokenizer, model = load_gen_model()
                    answer = generate_answer(prompt, tokenizer, model, max_new_tokens=300)

                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    answer = "Sorry — generation failed. See error above."
                st.session_state.chat_history.append({"role": "ai", "text": answer})

    # Render chat history
    for i, m in enumerate(st.session_state.chat_history):
        if m["role"] == "user":
            st.write(user_template.replace("{{MSG}}", m["text"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", m["text"]), unsafe_allow_html=True)

            # session refresh/ reload



if __name__ == '__main__':
    main()











