from langchain_core.vectorstores import VectorStoreRetriever, VectorStore
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List, Any


class CustomVectorStoreRetriever(VectorStoreRetriever):
    """Custom Retriever class that overrides the _get_relevant_documents method."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search_with_score(query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")

        # Custom logic for changing the output of the relevant documents
        
        return docs

def as_retriever(self, **kwargs: Any) -> CustomVectorStoreRetriever:
    """Return VectorStoreRetriever initialized from this VectorStore.

    Args:
        search_type (Optional[str]): Defines the type of search that
            the Retriever should perform.
            Can be "similarity" (default), "mmr", or
            "similarity_score_threshold".
        search_kwargs (Optional[Dict]): Keyword arguments to pass to the
            search function. Can include things like:
                k: Amount of documents to return (Default: 4)
                score_threshold: Minimum relevance threshold
                    for similarity_score_threshold
                fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
                lambda_mult: Diversity of results returned by MMR;
                    1 for minimum diversity and 0 for maximum. (Default: 0.5)
                filter: Filter by document metadata

    Returns:
        VectorStoreRetriever: Retriever class for VectorStore.

    Examples:

    .. code-block:: python

        # Retrieve more documents with higher diversity
        # Useful if your dataset has many similar documents
        docsearch.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 6, 'lambda_mult': 0.25}
        )

        # Fetch more documents for the MMR algorithm to consider
        # But only return the top 5
        docsearch.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 5, 'fetch_k': 50}
        )

        # Only retrieve documents that have a relevance score
        # Above a certain threshold
        docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.8}
        )

        # Only get the single most similar document from the dataset
        docsearch.as_retriever(search_kwargs={'k': 1})

        # Use a filter to only retrieve documents from a specific paper
        docsearch.as_retriever(
            search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
        )
    """
    tags = kwargs.pop("tags", None) or [] + self._get_retriever_tags()
    return CustomVectorStoreRetriever(vectorstore=self, tags=tags, **kwargs)

class CustomQDrant(Qdrant):
    pass

CustomQDrant.as_retriever=as_retriever