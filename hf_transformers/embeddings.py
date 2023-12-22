from langchain.embeddings import HuggingFaceBgeEmbeddings

model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)
print(len(model.embed_query("asdf")))
