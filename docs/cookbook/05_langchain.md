# LangChain Integration

`rustfuzz` provides a native `BaseRetriever` class that acts as a drop-in component for your LangChain `Runnable` pipelines.

Because `rustfuzz` executes `BM25Okapi` indices entirely in compiled machine code, this LangChain retriever runs orders of magnitude faster than standard `langchain_community` Python implementations. 

---

## Installation 

Make sure you have both libraries available:

```bash
pip install rustfuzz langchain-core
```

---

## Basic Retrieval Pipeline

You can instantiate `RustfuzzBM25Retriever` natively from strings or via `Document` objects.

```python
from langchain_core.documents import Document
from rustfuzz.langchain import RustfuzzBM25Retriever

# 1. Prepare LangChain Documents
docs = [
    Document(page_content="Apples and Oranges are fresh fruit.", metadata={"id": 1}),
    Document(page_content="The dog chased the cat down the street.", metadata={"id": 2}),
    Document(page_content="Apples are grown natively in Washington.", metadata={"id": 3})
]

# 2. Build the Rustfuzz Retriever 
# (BM25 is evaluated instantaneously here in Rust memory space)
retriever = RustfuzzBM25Retriever.from_documents(docs, k=2)

# 3. Retrieve
results = retriever.invoke("Apples")

for doc in results:
    print(doc.page_content)
    # The rustfuzz integration natively attaches the calculated BM25 float score 
    # to the `metadata["score"]` property of every returned Document.
    print(f"BM25 Score: {doc.metadata['score']}")
```

---

## Advanced Constructor (From Texts)

If you are just streaming text and do not have initialized `Document` objects, you can skip the boilerplate using `.from_texts`:

```python
texts = [
    "I love apples",
    "Oranges are decent"
]
metadatas = [{"source": "A"}, {"source": "B"}]

retriever = RustfuzzBM25Retriever.from_texts(
    texts, 
    metadatas=metadatas, 
    k=1, 
    k1=1.5,   # BM25 tuning parameters natively passed to the core index
    b=0.75
)

results = retriever.invoke("apples")
```

---

## Combining with RAG (Q&A Chain)

Since `RustfuzzBM25Retriever` conforms fully to LangChain's standard Protocol, you can seamlessly wire it into Conversational Retrieval Chains.

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_template("""
Answer the question uniquely based on the provided context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# The Rustfuzz Retriever acts as the data fetcher
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("Where are apples grown natively?")
```
