from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.retriever import load_retriever
from dotenv import load_dotenv

load_dotenv()

# Load retriever
retriever = load_retriever()

# LLM
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    temperature=0.5,
    max_output_tokens=512
)

# Prompt template
template = """
You are an AI assistant for legal documents. Use the context to answer the question.
If the answer is not found in the context, use your general legal knowledge to provide the best possible answer.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

def get_answer_stream(question, history=None):
    if history is None:
        history = []

    # Prepare conversation history text
    history_text = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    # Retrieve relevant docs for the *current* question
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.format(
        context=context,
        question=f"{history_text}\nUser: {question}"
    )

    for chunk in llm.stream(final_prompt):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content  # partial token

    yield {"docs": docs}