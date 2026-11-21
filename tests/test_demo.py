"""Offline checks for the RAG data flow and safe index reset behavior.

Run with: python -m unittest discover -s tests -v
No API credentials or external services are used.
Small synthetic Pinecone passages stand in for Medium article chunks, so the
tests are independent of the bundled text and do not evaluate answer quality.
"""

import os
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import ingestion
import main as demo


class RetrievalTests(unittest.TestCase):
    """Check that both implementations pass the question and retrieved context."""

    def setUp(self):
        self.question = "what is Pinecone in machine learning"
        self.docs = [
            Document(
                page_content="Pinecone is a managed vector database.",
                metadata={"source": "mediumblog1.txt"},
            ),
            Document(
                page_content="It supports similarity search over embeddings.",
                metadata={"source": "mediumblog1.txt"},
            ),
        ]
        self.answer = "Pinecone is a managed vector database for searching embeddings."

    def test_format_docs_handles_documents_and_empty_results(self):
        self.assertEqual(
            demo.format_docs(self.docs),
            "Pinecone is a managed vector database.\n\n"
            "It supports similarity search over embeddings.",
        )
        self.assertEqual(demo.format_docs([]), "")

    def test_manual_rag_passes_question_and_context(self):
        retriever = Mock()
        retriever.invoke.return_value = self.docs
        llm = Mock()
        llm.invoke.return_value = AIMessage(content=self.answer)

        answer = demo.retrieval_chain_without_lcel(self.question, retriever, llm)

        self.assertEqual(answer, self.answer)
        retriever.invoke.assert_called_once_with(self.question)
        messages = llm.invoke.call_args.args[0]
        content = messages[0].content
        self.assertIn(self.question, content)
        self.assertIn(demo.format_docs(self.docs), content)

    def test_lcel_chain_waits_for_invoke_and_returns_text(self):
        retrieved_questions = []

        def retrieve(question):
            retrieved_questions.append(question)
            return self.docs

        def respond(prompt):
            content = prompt.to_messages()[0].content
            self.assertIn(self.question, content)
            self.assertIn(demo.format_docs(self.docs), content)
            return AIMessage(content=self.answer)

        chain = demo.retrieval_chain_with_lcel(
            RunnableLambda(retrieve), RunnableLambda(respond)
        )
        self.assertEqual(retrieved_questions, [])
        self.assertEqual(chain.invoke({"question": self.question}), self.answer)
        self.assertEqual(retrieved_questions, [self.question])


class IngestionTests(unittest.TestCase):
    """Replace all remote clients with mocks before exercising ingestion."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "GOOGLE_API_KEY": "offline-google-placeholder",
                    "PINECONE_API_KEY": "offline-pinecone-placeholder",
                    "INDEX_NAME": "offline-test-index",
                },
                clear=True,
            )
        )
        self.stack.enter_context(patch.object(ingestion, "load_dotenv"))
        # Use predictable source text to check uploads independently of changes
        # to the bundled Medium article.
        self.source_text = (
            "Pinecone is a managed vector database.\n\n"
            "It supports similarity search over embeddings."
        )
        self.read_source = self.stack.enter_context(
            patch.object(
                ingestion.Path,
                "read_text",
                return_value=self.source_text,
            )
        )
        self.stack.enter_context(redirect_stdout(StringIO()))
        self.pc = self.stack.enter_context(patch.object(ingestion, "Pinecone"))
        self.embeddings = self.stack.enter_context(
            patch.object(ingestion, "GoogleGenerativeAIEmbeddings")
        )
        self.upload = self.stack.enter_context(
            patch.object(ingestion.PineconeVectorStore, "from_documents")
        )
        self.pc.return_value.has_index.return_value = True

    def test_existing_index_is_not_deleted_by_default(self):
        with self.assertRaisesRegex(SystemExit, "left unchanged"):
            ingestion.main()

        self.pc.return_value.delete_index.assert_not_called()
        self.pc.return_value.create_index.assert_not_called()
        self.upload.assert_not_called()

    def test_reset_is_explicit_and_uploads_source_text(self):
        ingestion.main(reset_index=True)

        self.pc.return_value.delete_index.assert_called_once_with(
            "offline-test-index", timeout=60
        )
        create_arguments = self.pc.return_value.create_index.call_args.kwargs
        self.assertEqual(create_arguments["dimension"], 1536)
        self.assertEqual(create_arguments["timeout"], 60)
        self.embeddings.assert_called_once_with(
            model="gemini-embedding-001", output_dimensionality=1536
        )
        self.upload.assert_called_once()
        chunks = self.upload.call_args.args[0]
        self.assertGreater(len(chunks), 0)
        self.assertEqual(
            "\n\n".join(doc.page_content for doc in chunks), self.source_text
        )
        self.assertTrue(
            all(doc.metadata["source"] == "mediumblog1.txt" for doc in chunks)
        )

    def test_new_index_needs_no_reset(self):
        self.pc.return_value.has_index.return_value = False

        ingestion.main()

        self.pc.return_value.delete_index.assert_not_called()
        self.pc.return_value.create_index.assert_called_once()
        self.upload.assert_called_once()

    def test_missing_configuration_fails_before_remote_calls(self):
        os.environ.pop("GOOGLE_API_KEY")

        with self.assertRaisesRegex(SystemExit, "GOOGLE_API_KEY"):
            ingestion.main()

        self.pc.assert_not_called()
        self.upload.assert_not_called()

    def test_empty_source_fails_before_reset(self):
        self.read_source.return_value = "  "
        with self.assertRaisesRegex(SystemExit, "mediumblog1.txt is empty"):
            ingestion.main(reset_index=True)

        self.pc.assert_not_called()
        self.upload.assert_not_called()

    def test_missing_source_fails_before_reset(self):
        self.read_source.side_effect = FileNotFoundError
        with self.assertRaisesRegex(SystemExit, "mediumblog1.txt was not found"):
            ingestion.main(reset_index=True)

        self.pc.assert_not_called()
        self.upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
