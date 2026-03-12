"""
Azure Embedding Providers Example for rustfuzz Retriever.

This example demonstrates how to use the high-level `Retriever` class
with models deployed on Azure AI:
1. Azure OpenAI (text-embedding-3-small)
2. Azure Cohere (embed-english-v3.0 via Azure AI Inference)

Prerequisites:
    uv add openai azure-ai-inference
"""

import os

from rustfuzz import Retriever


def main() -> None:
    # A tiny corpus for demonstration
    docs = [
        "Apple iPhone 15 Pro",
        "Samsung Galaxy S24 Ultra",
        "Sony PlayStation 5",
        "Microsoft Xbox Series X",
        "Nintendo Switch OLED",
    ]

    print("--- 1. Azure OpenAI ---")
    # To run this, you need AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY
    # environment variables set. You can also pass `api_key` and `api_base`
    # explicitly to the Retriever constructor.
    # We use a dummy key here just to show the initialization syntax.
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-openai-key"

    try:
        # "azure-openai" defaults to "text-embedding-3-small".
        # You can also use "azure-openai-large" for "text-embedding-3-large".
        retriever_aoai = Retriever(docs, embeddings="azure-openai")
        print(
            f"Initialized Azure OpenAI Retriever with {retriever_aoai.num_docs} documents!"
        )

        # In a real environment with a valid key, you could run:
        # results = retriever_aoai.search("gaming console", n=2)
        # print("Results:", results)
    except Exception as e:
        print(f"Failed or skipped Azure OpenAI initialization: {e}")

    print("\n--- 2. Azure Cohere ---")
    # To run this, you need AZURE_COHERE_ENDPOINT and AZURE_COHERE_API_KEY.
    # This refers to models deployed on Azure AI serverless endpoints natively.
    os.environ["AZURE_COHERE_ENDPOINT"] = (
        "https://your-cohere-endpoint.azureresources.com"
    )
    os.environ["AZURE_COHERE_API_KEY"] = "your-azure-cohere-key"

    try:
        # "azure-cohere" defaults to "embed-english-v3.0".
        # You can also use "azure-cohere-multilingual" for the multilingual version.
        retriever_acohere = Retriever(docs, embeddings="azure-cohere")
        print(
            f"Initialized Azure Cohere Retriever with {retriever_acohere.num_docs} documents!"
        )

        # In a real environment with a valid key, you could run:
        # results = retriever_acohere.search("gaming console", n=2)
        # print("Results:", results)
    except Exception as e:
        print(f"Failed or skipped Azure Cohere initialization: {e}")


if __name__ == "__main__":
    main()
