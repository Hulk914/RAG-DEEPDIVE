# RAG mini demo with Gemini and Pinecone

A small, readable project for learning **Retrieval-Augmented Generation (RAG)**
and **LangChain Expression Language (LCEL)**. It asks the same question three ways:

| Implementation | What happens |
| --- | --- |
| 0: Without RAG | Send only the question to Gemini. |
| 1: RAG without LCEL | Retrieve documents, build the prompt, and invoke Gemini through explicit Python calls. |
| 2: RAG with LCEL | Compose the same retrieval, prompt, model, and output steps using the pipe operator. |

RAG supplies external information to the model. LCEL connects the steps of the
workflow. Both RAG implementations use LangChain components.

## What you will learn

- How text becomes chunks, embeddings, and searchable Pinecone records.
- How retrieved text becomes context in a model's prompt.
- How an explicit Python workflow maps to an LCEL chain.
- The benefits and drawbacks of retrieval and of each composition style.

This is a learning demo rather than a benchmark or a production application.

## Project files

| File | Purpose |
| --- | --- |
| `ingestion.py` | Read the Medium article text, split it, embed the chunks, and upload them to Pinecone. |
| `main.py` | Run the three answer implementations using the same question. |
| `mediumblog1.txt` | Included Medium article text used as the demo knowledge base. |
| `.env.example` | Placeholder configuration to copy locally; contains no real credentials. |
| `.gitignore` | Exclude local secrets, virtual environments, and caches from Git. |
| `tests/test_demo.py` | Offline checks using fake model/retriever responses and mocked Pinecone calls. |
| `pyproject.toml` / `uv.lock` | Dependencies and the locked versions used by the project. |

## Set up and run

You need Python 3.13, [uv](https://docs.astral.sh/uv/getting-started/installation/),
a [Google Gemini API key](https://aistudio.google.com/apikey), and a
[Pinecone account and API key](https://app.pinecone.io/). The configured models and
AWS `us-east-1` serverless deployment must be available to your accounts.

### 1. Install dependencies

Clone or download this repository. Open a terminal in the directory containing
`main.py`, then run:

```powershell
uv sync --locked
```

This creates the local Python environment from `uv.lock`. No OpenAI key, external
document parser, or downloaded language model is required. Formatting tools are
in the development dependency group; the demo has six direct runtime dependencies.

### 2. Create your local configuration

In PowerShell:

```powershell
Copy-Item .env.example .env
```

On macOS or Linux, use `cp .env.example .env`. If you already have `.env`,
edit it instead of copying over it.

Open `.env` and replace the two API-key placeholders. Set `INDEX_NAME` to a
dedicated demo index name, such as `rag-mini-demo`, using lowercase letters,
numbers, and hyphens. You do not need to create the index manually.

The chat model defaults to `gemini-3.5-flash-lite`. The optional `GEMINI_MODEL`
setting lets you select another Gemini chat model available to your API account.
Both scripts use `gemini-embedding-001` with 1,536 dimensions for embeddings.

### 3. Ingest the included Medium example

The repository includes `mediumblog1.txt` next to `ingestion.py`, ready for
ingestion after you configure `.env`. The text comes from **Vector Database:
What is it and why you should know it?** by Ejiro Onose; see the
[original Medium article](https://medium.com/@EjiroOnose/vector-database-what-is-it-and-why-you-should-know-it-ae7e7dca82a4).

```powershell
uv run --locked python ingestion.py
```

This reads `mediumblog1.txt`, creates a Pinecone index, and uploads the embedded chunks.
Paths are resolved relative to the script, so there is no machine-specific path
to edit. Index creation waits up to 60 seconds for readiness. Uploaded records
can take a few more seconds to become searchable.

If that index already exists, ingestion stops and leaves it unchanged. To rebuild
your dedicated demo index after editing the source text, run:

```powershell
uv run --locked python ingestion.py --reset-index
```

**`--reset-index` deletes the named index and everything in it before recreating
it.** Alternatively, choose a new `INDEX_NAME` in `.env`.

### 4. Compare the three answers

```powershell
uv run --locked python main.py
```

The default question is:

> what is Pinecone in machine learning

The RAG versions retrieve relevant parts of the Medium article before answering.
The direct-answer baseline receives only the question. Since Pinecone is a
general topic, the direct model may already answer well; compare how the RAG
answers use the article's context. Exact responses vary, and this comparison
does not measure accuracy improvements.

You will see three labeled sections, one per implementation. A complete run
makes three chat-model calls and performs embedding/search separately for each
RAG version. These API operations and Pinecone resources can incur charges.
`main.py` initializes shared clients even for the direct-answer baseline.

## Experiment with your own text

Replace the contents of `mediumblog1.txt` and change `query` inside `main()` to a
question about your text.
Rebuild the dedicated index, then run the comparison again. Try a question whose
answer is absent from the source too.

The text is sent to Google's embedding API and stored in Pinecone; the retrieved
chunks are sent to the Gemini chat model. Use sample material you are comfortable
sending to those services and have permission to share.

Keep the embedding model, vector size, and index name consistent across both
scripts. If you change the embedding configuration, rebuild the index as well.
RAG updates the information in the prompt; it does not retrain the chat model.

## How it works

```text
Ingestion:
mediumblog1.txt -> Document -> chunks -> embedding model -> Pinecone

RAG at question time:
question -> embed and search Pinecone -> retrieved chunks
         -> context + question -> Gemini -> answer
```

An embedding is a list of numbers used to compare text meaning. The embedding
model supports retrieval; the chat model separately writes the answer.

The splitter targets 1,000 characters per chunk with zero overlap. It splits at
blank lines; long paragraphs can exceed that target. Pinecone uses a dense index
with cosine similarity. The retriever requests up to three chunks (`k=3`).

### Explicit Python version

`retrieval_chain_without_lcel(query, retriever, llm)` performs these steps:

1. `retriever.invoke(query)` retrieves documents.
2. `format_docs(docs)` joins their text with blank lines.
3. `prompt_template.format_messages(...)` inserts the context and question.
4. `llm.invoke(messages)` generates a response, whose content is returned.

### LCEL version

`retrieval_chain_with_lcel(retriever, llm)` builds:

```python
RunnablePassthrough.assign(
    context=itemgetter("question") | retriever | format_docs
) | prompt_template | llm | StrOutputParser()
```

- `itemgetter("question")` extracts the question from the input dictionary.
- `retriever | format_docs` turns that question into retrieved context text.
- `assign(context=...)` keeps the original input fields and adds the context.
- `prompt_template` fills `{question}` and `{context}`.
- `llm` writes the answer; `StrOutputParser()` returns plain text.

Building the chain prepares the workflow. Calling
`.invoke({"question": query})` executes it. See the
[RunnablePassthrough reference](https://reference.langchain.com/python/langchain-core/runnables/passthrough/RunnablePassthrough)
for how `assign()` merges fields into the input dictionary.

## RAG versus a direct LLM call

These are practical tradeoffs for this demo. The
[LangChain retrieval guide](https://docs.langchain.com/oss/python/deepagents/retrieval)
explains how external knowledge becomes context for generation.

| Approach | Benefits | Drawbacks |
| --- | --- | --- |
| Without RAG | Simpler answer flow; fewer network steps and prompt tokens; useful for general questions. | Cannot consult the project's source text; knowledge can be outdated; answers can be invented or ungrounded. |
| With RAG | Can use your documents and newly indexed information without retraining; relevant context can improve grounding. | Requires ingestion, storage, and retrieval; adds latency and cost; source quality, chunking, and search affect results; incorrect answers remain possible. |

RAG is only as current as the indexed source. The prompt asks the model to say it
does not know when context is insufficient, but this instruction is not an
enforced guarantee. The demo has no relevance threshold, automatic citations,
evaluation dataset, reranking, conversation memory, or agent loop.

## LCEL versus explicit Python calls

| Approach | Benefits | Drawbacks |
| --- | --- | --- |
| Without LCEL | Familiar Python; visible intermediate values; straightforward breakpoints and custom control flow. | More wiring as workflows grow; a whole-workflow async, batch, or streaming interface requires additional implementation. |
| With LCEL | Compact, reusable composition and a common chain interface for invocation, async execution, batching, and streaming where components support it. | Requires understanding runnables and input/output shapes; intermediate values are less visible; custom branching can be clearer in ordinary Python. |

The [RunnableSequence reference](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence)
describes these interfaces and streaming requirements. This demo only uses
synchronous `.invoke()`. LCEL does not automatically improve answer quality or
make a single question faster; the two RAG versions use the same prompt and
retrieval settings.

## Offline checks

```powershell
uv run --locked python -m unittest discover -s tests -v
```

These checks need no credentials or network access once dependencies are installed.
Small synthetic Pinecone passages stand in for retrieved article chunks, keeping
the checks predictable and independent of changes to the bundled article.
They check question/context propagation, text output, missing configuration,
missing or empty source files, and protection against accidental index deletion.
They do not validate live Gemini access, Pinecone provisioning, or answer quality.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Missing configuration | Copy `.env.example` to `.env` and fill in the required values. |
| Missing source file | Restore the included `mediumblog1.txt` from this repository next to `ingestion.py`. |
| Authentication or model-access error | Check the relevant API key and account access. Set `GEMINI_MODEL` to an available chat model if needed. |
| Quota or rate-limit error | Check the provider's quota/billing settings and retry after the limit resets. |
| Index not found | Run ingestion first and use the same `INDEX_NAME` for both scripts. |
| Index already exists | Choose a fresh demo index name or explicitly rebuild with `--reset-index`. |
| Index creation times out | Check its status in Pinecone before retrying. A timed-out creation may still finish. |
| Empty or unexpected context | Wait briefly after upload, check that the index contains records, and ask about `mediumblog1.txt`. |
| Vector dimension mismatch | Use 1,536 dimensions for this demo and rebuild the dedicated index after embedding changes. |

## Before publishing to GitHub

`.gitignore` excludes `.env`, local `.env.*` variants, private-key files, virtual
environments, and caches. `.env.example` intentionally remains publishable and
must contain placeholders only. Keep `uv.lock` in the repository.

Check the files Git will publish:

```powershell
git check-ignore .env
git status --short
```

The first command should print `.env`; it should not appear as a new file in
the second command. Review the other files before committing, including output
or screenshots you plan to share. Git ignore rules do not remove secrets already
committed; previously exposed keys must be revoked or rotated. See the
[Git ignore documentation](https://git-scm.com/docs/gitignore).
