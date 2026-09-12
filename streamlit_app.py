import json
import os
import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prevent OpenMP runtime conflicts on Windows systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================================================================
# 1. Page Configuration & Header
# ==============================================================================
st.set_page_config(
    page_title="LoreKeeper | Agentic GraphRAG Studio",
    page_icon="📜",
    layout="wide",
)

st.title("📜 LoreKeeper: Universal Narrative Continuity & 'What-If' Studio")
st.markdown(
    "*Powered by local Open-Source Deep Learning (`Qwen2.5-1.5B` + LoRA on CUDA) & Dynamic GraphRAG.*"
)


# ==============================================================================
# 2. Resource Caching: Load Models Once into GPU VRAM
# ==============================================================================
@st.cache_resource
def load_lorekeeper_engines():
    """Loads BGE-small embeddings and fine-tuned Qwen-1.5B into GPU memory once."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load English Embedding Model
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    # 2. Load Base LLM and Tokenizer
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_path = "./lorekeeper_lora_weights"

    tokenizer = AutoTokenizer.from_pretrained(
        lora_path if os.path.exists(lora_path) else base_model_id
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_llm = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    # 3. Attach LoRA weights if available, else fallback to base model
    if os.path.exists(lora_path):
        llm = PeftModel.from_pretrained(base_llm, lora_path)
        status_msg = "LoRA Fine-Tuned Model (Active)"
    else:
        llm = base_llm
        status_msg = "Base Model (LoRA weights not found)"

    llm.eval()
    return embed_model, llm, tokenizer, device, status_msg


with st.spinner("Initializing neural engines on GPU..."):
    embed_model, local_llm, tokenizer, device, model_status = (
        load_lorekeeper_engines()
    )

st.sidebar.markdown(f"**Hardware**: `{device.upper()}`")
st.sidebar.markdown(f"**Model Status**: `{model_status}`")


# ==============================================================================
# 3. Dynamic RAG & Knowledge Graph Engine
# ==============================================================================
def chunk_story_text(raw_text, target_words=200, overlap=25):
    """Dynamically slices arbitrary text into cohesive 200-word passages."""
    words = raw_text.split()
    chunks = []
    step = target_words - overlap
    for i in range(0, len(words), step):
        c = " ".join(words[i : i + target_words])
        if len(c.split()) >= 30:
            chunks.append(c)
    return chunks


def build_dynamic_knowledge_graph(story_text):
    """Extracts prominent character entities and co-occurrence relations from text."""
    kg = nx.DiGraph()

    # Heuristic regex: Extract capitalized multi-word entity mentions
    raw_candidates = list(
        set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", story_text))
    )
    # Filter common stopwords and keep frequent proper nouns
    stopwords = {"The", "This", "That", "Chapter", "Book", "When", "There", "Here"}
    entities = [
        e
        for e in raw_candidates
        if e not in stopwords and story_text.count(e) >= 2 and len(e.split()) <= 2
    ][:12]

    for ent in entities:
        kg.add_node(ent, type="Character", mentions=story_text.count(ent))

    # Form edges based on paragraph co-occurrence
    paragraphs = [p for p in story_text.split("\n\n") if len(p.strip()) > 0]
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            e1, e2 = entities[i], entities[j]
            co_count = sum(1 for p in paragraphs if e1 in p and e2 in p)
            if co_count >= 1:
                kg.add_edge(e1, e2, relation="CO_OCCURS_WITH", weight=co_count)

    return kg, entities


class DynamicRetriever:
    """NumPy-native cosine similarity retriever for uploaded story chunks."""

    def __init__(self, chunks, vectors, model):
        self.chunks = chunks
        self.vectors = vectors
        self.model = model

    def search(self, query, top_k=2):
        q_vec = self.model.encode([query], normalize_embeddings=True).squeeze()
        sims = np.dot(self.vectors, q_vec)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[idx], float(sims[idx])) for idx in top_idx]


# ==============================================================================
# 4. Sidebar: User Ingestion
# ==============================================================================
st.sidebar.header("📁 Step 1: Upload Story")
uploaded_file = st.sidebar.file_uploader(
    "Upload story text (.txt / .md)", type=["txt", "md"]
)

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if uploaded_file is not None and not st.session_state.indexed:
    story_raw = uploaded_file.read().decode("utf-8", errors="ignore")
    st.session_state.story_text = story_raw

    with st.spinner("Processing story into GraphRAG memory..."):
        # 1. Chunking
        chunks = chunk_story_text(story_raw)
        st.session_state.chunks = chunks

        # 2. Compute embeddings
        vectors = embed_model.encode(
            chunks, batch_size=16, normalize_embeddings=True, show_progress_bar=False
        )
        st.session_state.retriever = DynamicRetriever(
            chunks, vectors, embed_model
        )

        # 3. Dynamic Knowledge Graph
        kg, entities = build_dynamic_knowledge_graph(story_raw)
        st.session_state.kg = kg
        st.session_state.entities = entities
        st.session_state.indexed = True

    st.sidebar.success(
        f"✔ Ingested: {len(chunks)} Chunks | {len(entities)} Detected Entities"
    )

if st.sidebar.button("Clear / Upload New Story"):
    st.session_state.indexed = False
    st.rerun()


# ==============================================================================
# 5. Main Functional Interface
# ==============================================================================
if st.session_state.indexed:
    tab1, tab2, tab3 = st.tabs(
        [
            "🔍 Tab 1: Plot-Hole Scanner",
            "🦋 Tab 2: 'What-If' Simulation",
            "📊 Tab 3: World Knowledge Graph",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: PLOT-HOLE SCANNER
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Automated Plot-Hole & Continuity Checker")
        st.write(
            "Audit a newly written scene or chapter against the canonical context of the uploaded story."
        )

        draft_scene = st.text_area(
            "Draft Scene to Audit:",
            height=120,
            placeholder="e.g., The hero pulled out his magic golden staff and teleported across the mountain...",
        )

        selected_char = st.selectbox(
            "Select Principal Character involved:",
            st.session_state.entities
            if st.session_state.entities
            else ["Protagonist"],
        )

        if st.button("Run Continuity Audit"):
            if draft_scene.strip():
                with st.spinner(
                    "Auditing scene against Graph facts and narrative contexts..."
                ):
                    # 1. Retrieve narrative passages
                    search_results = st.session_state.retriever.search(
                        f"{selected_char} {draft_scene}", top_k=2
                    )
                    narrative_context = "\n\n".join(
                        [f"- {doc}" for doc, _ in search_results]
                    )

                    # 2. Retrieve Graph edges
                    graph_neighbors = []
                    if selected_char in st.session_state.kg:
                        for _, v, d in st.session_state.kg.out_edges(
                            selected_char, data=True
                        ):
                            graph_neighbors.append(f"{d['relation']} -> {v}")
                    graph_facts = (
                        ", ".join(graph_neighbors)
                        if graph_neighbors
                        else "No direct relational constraints found."
                    )

                    # 3. Format Prompt (Aligned with LoRA Training Format!)
                    audit_prompt = (
                        f"Facts: Character: {selected_char} | Relational Context: {graph_facts}\n"
                        f'Draft: "{draft_scene}"\n'
                        f"Analyze consistency and conclude strictly with [PLOT-HOLE] or [CONSISTENT]."
                    )

                    messages = [
                        {
                            "role": "system",
                            "content": "You are a precise literary continuity auditor.",
                        },
                        {"role": "user", "content": audit_prompt},
                    ]

                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    tokens = tokenizer([formatted], return_tensors="pt").to(
                        device
                    )

                    with torch.no_grad():
                        out = local_llm.generate(
                            **tokens, max_new_tokens=150, do_sample=False
                        )

                    verdict_text = tokenizer.decode(
                        out[0][len(tokens.input_ids[0]) :],
                        skip_special_tokens=True,
                    )

                    # 4. Display Judgement
                    if "[PLOT-HOLE]" in verdict_text.upper():
                        st.error("🚨 **Verdict: [PLOT-HOLE DETECTED]**")
                    else:
                        st.success("✅ **Verdict: [CONSISTENT WITH CANON]**")

                    st.markdown(f"**Audit Reasoning:**\n\n{verdict_text}")

                    with st.expander("Inspected Story Lore (Retrieved Passages)"):
                        st.write(narrative_context)
            else:
                st.warning("Please provide a draft scene to audit.")

    # -------------------------------------------------------------
    # TAB 2: 'WHAT-IF' SIMULATION ENGINE
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Branching Counterfactual 'What-If' Simulation")
        st.write(
            "Input a counterfactual hypothesis to simulate an alternate parallel timeline."
        )

        what_if_input = st.text_input(
            "Enter 'What-If' Premise:",
            placeholder="e.g., What if the mentor refused to give the hero the sacred key?",
        )

        col1, col2 = st.columns(2)
        with col1:
            temp_val = st.slider("Creativity (Temperature)", 0.3, 1.0, 0.7, 0.1)
        with col2:
            max_tokens = st.slider(
                "Generation Length (Tokens)", 300, 800, 500, 50
            )

        if st.button("Extrapolate Parallel Universe"):
            if what_if_input.strip():
                with st.spinner("Extrapolating Butterfly Effect on GPU..."):
                    # Grounding lore retrieval
                    ground_docs = st.session_state.retriever.search(
                        what_if_input, top_k=2
                    )
                    ground_context = "\n".join([d[0][:300] for d, _ in ground_docs])

                    cf_prompt = f"""You are the LoreKeeper Butterfly Effect Simulation Engine.
Extrapolate a counterfactual branching timeline based on the premise while preserving core world rules.

【Ground Story Context】:
{ground_context}

【Counterfactual Premise】:
"{what_if_input}"

TASK:
Extrapolate the timeline in 3 structured phases:
1. Immediate Reaction (Direct aftermath of this divergence).
2. Relationship Realignment (How character alliances and dynamics alter).
3. Long-term Narrative Divergence (Altered climax and destination).
"""
                    messages = [
                        {
                            "role": "system",
                            "content": "You are an expert causal narrative extrapolation engine.",
                        },
                        {"role": "user", "content": cf_prompt},
                    ]
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    tokens = tokenizer([formatted], return_tensors="pt").to(
                        device
                    )

                    with torch.no_grad():
                        out = local_llm.generate(
                            **tokens,
                            max_new_tokens=max_tokens,
                            temperature=temp_val,
                            top_p=0.9,
                            do_sample=True,
                        )

                    simulation_story = tokenizer.decode(
                        out[0][len(tokens.input_ids[0]) :],
                        skip_special_tokens=True,
                    )

                    st.markdown("### 🌌 Extrapolated Parallel Timeline")
                    st.info(simulation_story)

                    # Minimal edits retention
                    story_words = set(simulation_story.lower().split())
                    ref_words = set(ground_context.lower().split())
                    invariance_score = len(
                        story_words.intersection(ref_words)
                    ) / max(len(story_words), 1)

                    st.caption(
                        f"📊 **Causal Invariance Retention**: `{invariance_score:.4f}` (Preservation of world vocabulary)"
                    )
            else:
                st.warning("Please enter a 'What-If' premise.")

    # -------------------------------------------------------------
    # TAB 3: DYNAMIC KNOWLEDGE GRAPH
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Dynamic World Entities & Graph Topology")
        st.write(
            f"Extracted **{len(st.session_state.entities)}** key entities and **{st.session_state.kg.number_of_edges()}** relationships from the uploaded story."
        )

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### Key Entities Extracted")
            for ent in st.session_state.entities:
                st.markdown(f"- **{ent}**")

        with col_right:
            st.markdown("#### Graph Visualization")
            if st.session_state.kg.number_of_edges() > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                pos = nx.spring_layout(st.session_state.kg, seed=42)
                nx.draw_networkx(
                    st.session_state.kg,
                    pos,
                    ax=ax,
                    with_labels=True,
                    node_color="skyblue",
                    node_size=1200,
                    font_size=8,
                    font_weight="bold",
                    edge_color="gray",
                )
                plt.axis("off")
                st.pyplot(fig)
            else:
                st.info(
                    "Entities detected, but paragraphs lacked direct co-occurrence edges."
                )

        with st.expander("Inspect Raw Semantic Chunks (First 3 Chunks)"):
            for idx, c in enumerate(st.session_state.chunks[:3]):
                st.markdown(f"**[Chunk #{idx}]:** {c}")

else:
    st.info("👈 Please upload a story text file (.txt / .md) in the sidebar to begin!")