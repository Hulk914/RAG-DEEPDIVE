"""Compare direct Gemini answers with explicit Python RAG and LCEL RAG.

Run ingestion.py first, then run this script. All three implementations use the
same question and chat model. The two RAG versions also share the same retriever
and prompt, so their main difference is how the steps are connected.
"""

import os
from operator import itemgetter
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Both RAG implementations fill the same context and question placeholders.
prompt_template = ChatPromptTemplate.from_template("""
Answer the question based on the following context.
If the context does not contain the answer, say you do not know.

Context:
{context}

Question: {question}

Provide a detailed answer:
""")


def format_docs(docs):
    """Join retrieved Document.page_content values into one context string.

    Blank lines separate chunks. Metadata is omitted, so this helper does not
    add source citations. An empty result becomes an empty string.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str, retriever, llm):
    """Run RAG through explicit Python calls and return the response content.

    This still uses LangChain clients. Passing the retriever and model explicitly
    makes each dependency visible and lets us check the flow with offline fakes.
    """
    # 1. Search Pinecone using the question; this also embeds the query.
    docs = retriever.invoke(query)

    # 2. Convert the retrieved documents into text for the prompt.
    context = format_docs(docs)

    # 3. Insert the context and question into the shared chat prompt.
    messages = prompt_template.format_messages(context=context, question=query)

    # 4. Ask Gemini to answer using that context.
    response = llm.invoke(messages)
    return response.content


def retrieval_chain_with_lcel(retriever, llm):
    """Build a reusable chain accepting {"question": ...} and returning text.

    LCEL means LangChain Expression Language. The | operator passes each step's
    output to the next. Work starts when .invoke() is called on the built chain.
    """
    return (
        # Preserve the question and add context:
        # input["question"] -> retrieved Documents -> joined document text.
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        # The resulting dictionary supplies both prompt placeholders.
        | prompt_template
        | llm
        # Extract the model's answer as a plain string.
        | StrOutputParser()
    )


def main():
    """Initialize shared clients and print the three answers for comparison."""
    # Resolve .env relative to this file, regardless of the working directory.
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    required = ("GOOGLE_API_KEY", "PINECONE_API_KEY", "INDEX_NAME")
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        raise SystemExit(
            f"Missing configuration: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your credentials."
        )

    print("Initializing components...")
    # Match ingestion.py: the same model and vector size must be used for
    # document embeddings and query embeddings.
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=1536,
    )
    # The embedding model supports search; this chat model writes answers.
    llm = ChatGoogleGenerativeAI(
        temperature=0,
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        vertexai=False,
    )
    vector_store = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    # Return up to three relevant chunks from the already populated index.
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # This question concerns the supplied sample, so the direct-answer baseline
    # has no project notes to consult. Change it to explore other questions.
    query = (
        "Which embedding model, vector size, and retrieval limit does this demo use?"
    )

    # Implementation 0: no document retrieval, only the question.
    print("\n" + "=" * 70)
    print("Implementation 0: Without RAG")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print(result_raw.content)

    # Implementation 1: retrieve, format, and generate through explicit calls.
    print("\n" + "=" * 70)
    print("Implementation 1: RAG without LCEL")
    print("=" * 70)
    print(retrieval_chain_without_lcel(query, retriever, llm))

    # Implementation 2: run the same RAG workflow as a composed LCEL chain.
    print("\n" + "=" * 70)
    print("Implementation 2: RAG with LCEL")
    print("=" * 70)
    # assign() expects a dictionary containing the question key.
    chain = retrieval_chain_with_lcel(retriever, llm)
    print(chain.invoke({"question": query}))


if __name__ == "__main__":
    main()
