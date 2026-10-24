
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="Research Intelligence Platform",
    page_icon="📚",
    layout="wide",
)

st.title("Research Intelligence Platform")
st.caption("Explore research papers using semantic search and AI-powered answers.")

with st.sidebar:
    st.header("Search settings")
    top_k = st.slider(
        "Retrieved passages",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of relevant research passages provided to the LLM.",
    )
    st.divider()
    st.markdown("**Pipeline**")
    st.caption("Research papers → FAISS retrieval → Groq LLM")
    st.caption("API: `127.0.0.1:8000`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View retrieved sources"):
                for i, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**Source {i}: {source['source_file']}**")
                    st.write(f"Page: {source['page_number']}")
                    st.write(f"Category: {source['category']}")
                    st.write(f"Similarity score: {source['score']:.3f}")
                    if i < len(message["sources"]):
                        st.divider()

question = st.chat_input("Ask a question about your research papers...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and generating an answer..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question, "top_k": top_k},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("View retrieved sources", expanded=True):
                        for i, source in enumerate(sources, start=1):
                            with st.container(border=True):
                                st.markdown(f"**[Source {i}]**")
                                st.caption(source["source_file"])
                                st.write(
                                    f"Page: {source['page_number']} · "
                                    f"Category: {source['category']} · "
                                    f"Similarity: {source['score']:.3f}"
                                )
                            if i < len(sources):
                                st.divider()
                else:
                    st.info("No source metadata was returned.")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the API. Start FastAPI first using "
                    "`python -m uvicorn api.main:app --reload`."
                )
            except requests.exceptions.Timeout:
                st.error("The request timed out. Try again.")
            except requests.exceptions.HTTPError as exc:
                detail = exc.response.text[:1000] if exc.response else str(exc)
                st.error(f"API request failed: {detail}")
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f"Unexpected API response: {exc}")
