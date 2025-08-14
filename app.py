import streamlit as st
from src.rag_pipeline import get_answer_stream

st.set_page_config(page_title="Legal RAG Chatbot", layout="wide")


# Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = []  
if "history" not in st.session_state:
    st.session_state["history"] = []   


# Sidebar
st.sidebar.title("Chat History")

# Show previous chats
if st.session_state["history"]:
    for idx, chat in enumerate(st.session_state["history"], start=1):
        if st.sidebar.button(f"Chat {idx}: {chat['title'][:20]}..."):
            st.session_state["messages"] = chat["messages"].copy()

# Reset current chat
if st.sidebar.button("🗑️ Reset Chat"):
    if st.session_state["messages"]:
        # Save current chat into history
        st.session_state["history"].append({
            "title": st.session_state["messages"][0]["content"] if st.session_state["messages"] else "Untitled",
            "messages": st.session_state["messages"].copy()
        })
    st.session_state["messages"] = []

st.sidebar.markdown("---")
st.sidebar.write(f"Total Saved Chats: {len(st.session_state['history'])}")


# Chat Title
st.title("RAG Chatbot")


# Display Messages
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])


# User Input
user_input = st.chat_input("Ask a question about the document...")
if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Bot response
    with st.chat_message("assistant"):
        response_text = ""
        docs = []
        response_placeholder = st.empty()

        # for token in get_answer_stream(user_input):
        for token in get_answer_stream(user_input, history=st.session_state["messages"]):
            if isinstance(token, dict) and "docs" in token:
                docs = token["docs"]
            else:
                response_text += token
                response_placeholder.markdown(response_text)

        st.session_state["messages"].append({"role": "assistant", "content": response_text})

        if docs:
            with st.expander("Source Chunks"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"Chunk {i}: {d.page_content}")
