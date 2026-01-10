"""
Multi-hop RAG Retrieval using LlamaIndex with Local Llama3-8B Model

This module implements a knowledge graph-based multi-hop retrieval system that:
1. Loads and indexes documents using LlamaIndex
2. Constructs a knowledge graph with relationships between entities
3. Performs multi-hop retrieval (2-hop) for complex queries
4. Uses local Llama3-8B model for reasoning and answer generation
"""

import json
import re
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from llama_index.core import (
    VectorStoreIndex, 
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.core.schema import TextNode, Document, NodeWithScore
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
import os


@dataclass
class RetrievalResult:
    """Result from a single retrieval step."""
    query: str
    retrieved_nodes: List[NodeWithScore]
    retrieval_time: float
    hop_number: int


@dataclass 
class MultiHopRAGResult:
    """Result from multi-hop RAG retrieval."""
    original_query: str
    sub_queries: List[str]
    intermediate_results: List[RetrievalResult]
    final_answer: str
    total_retrieval_time: float
    total_generation_time: float
    all_retrieved_context: str


class MultiHopRetriever:
    """
    Multi-hop retriever that performs iterative retrieval and reasoning.
    
    This retriever:
    1. Analyzes the query to identify information needs
    2. Performs first-hop retrieval to get initial context
    3. Generates a sub-query based on initial results
    4. Performs second-hop retrieval to gather additional context
    5. Combines all retrieved information for final answer generation
    """
    
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        kg_index: Optional[KnowledgeGraphIndex] = None,
        top_k: int = 5,
        device: str = "cuda",
        tokenizer=None,
        model=None
    ):
        self.vector_index = vector_index
        self.kg_index = kg_index
        self.top_k = top_k
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        
        self.vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=top_k
        )
        
    
    def analyze_query(self, query: str) -> List[str]:
        """
        Analyze query to identify information needs.
        Returns list of sub-queries for multi-hop retrieval.
        """
        analysis_prompt = f"""
        Analyze the following question and break it down into sub-questions needed to answer it.
        For multi-hop reasoning, identify what intermediate information is needed.
        
        Question: {query}
        
        Return a JSON list of sub-questions. Example:
        ["What is X?", "What is the relationship between X and Y?"]
        """
        return [query]
    
    def first_hop_retrieve(self, query: str) -> RetrievalResult:
        """Perform first hop of retrieval."""
        start_time = time.perf_counter()
        
        retrieved_nodes = self.vector_retriever.retrieve(query)
        
        retrieval_time = time.perf_counter() - start_time
        
        return RetrievalResult(
            query=query,
            retrieved_nodes=retrieved_nodes,
            retrieval_time=retrieval_time,
            hop_number=1
        )
    
    def generate_sub_query(
        self, 
        original_query: str, 
        first_hop_results: List[NodeWithScore]
    ) -> str:
        """
        Generate a sub-query for the second hop based on first hop results.
        Uses LLM to intelligently generate a follow-up question.
        """
        context = "\n".join([node.get_text() for node in first_hop_results[:3]])
        
        sub_query_prompt = f"""<|start_header_id|>user<|end_header_id|>
You need to generate a follow-up question to find additional information for answering the original question.

Original Question: {original_query}

Retrieved Context:
{context}

Based on the context above, what specific additional information would help answer the original question?
Return only the follow-up question, nothing else.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
        
        inputs = self.tokenizer(sub_query_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.autograd.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split(
                "<|start_header_id|>assistant<|end_header_id|>"
            )[-1].strip()
        
        response = response.strip().strip('?').strip()
        if not response.endswith('?'):
            response = response + '?'
        
        print(f"[debug] request for generating sub query: \n{sub_query_prompt}\n response: \n{response}\n")
        return response
    
    def second_hop_retrieve(
        self, 
        sub_query: str,
        first_hop_context: str
    ) -> RetrievalResult:
        """Perform second hop of retrieval."""
        start_time = time.perf_counter()
        
        context_header = f"Context from first hop:\n{first_hop_context}"
        enhanced_query = f"{sub_query}\n\n{context_header}"
        retrieved_nodes = self.vector_retriever.retrieve(enhanced_query)
        
        retrieval_time = time.perf_counter() - start_time
        
        return RetrievalResult(
            query=sub_query,
            retrieved_nodes=retrieved_nodes,
            retrieval_time=retrieval_time,
            hop_number=2
        )
    
    def multi_hop_retrieve(self, query: str) -> Tuple[List[RetrievalResult], str]:
        """
        Perform multi-hop retrieval and return combined context.
        """
        results = []
        
        first_hop = self.first_hop_retrieve(query)
        results.append(first_hop)
        
        first_hop_context = "\n".join([
            node.get_text() for node in first_hop.retrieved_nodes
        ])
        
        sub_query = self.generate_sub_query(query, first_hop.retrieved_nodes)
        
        second_hop = self.second_hop_retrieve(sub_query, first_hop_context)
        results.append(second_hop)
        
        second_hop_text = "\n".join([node.get_text() for node in second_hop.retrieved_nodes])
        combined_context = f"""
        === First Hop Retrieval ===
        {first_hop_context}
        
        === Second Hop Retrieval ===
        {second_hop_text}
        """
        
        return results, combined_context


class LlamaIndexMultiHopRAG:
    """
    Complete Multi-hop RAG System using LlamaIndex with local Llama3-8B.
    
    This class integrates:
    - Document loading and indexing with LlamaIndex
    - Multi-hop retrieval strategy
    - Local LLM inference for reasoning
    """
    
    def __init__(
        self,
        model_path: str,
        embedding_model_path: str,
        documents: Optional[List[Document]] = None,
        persist_dir: Optional[str] = None,
        device: str = "cuda",
        top_k: int = 5
    ):
        self.model_path = model_path
        self.embedding_model_path = embedding_model_path
        self.device = device
        self.top_k = top_k
        
        self._setup_llm()
        self._setup_embeddings()
        self._setup_index(documents, persist_dir)
    
    def _setup_llm(self):
        """Setup local Llama3-8B model using HuggingFaceLLM."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"Loading Llama3-8B model from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            use_fast=False
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        
        print("Llama3-8B model loaded successfully")
    
    def _setup_embeddings(self):
        """Setup embedding model for retrieval."""
        print(f"Loading embedding model from: {self.embedding_model_path}")
        
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_path,
            device=self.device
        )
        
        Settings.embed_model = self.embed_model
        print("Embedding model loaded successfully")
    
    def _setup_index(
        self, 
        documents: Optional[List[Document]], 
        persist_dir: Optional[str]
    ):
        """Setup vector index for retrieval."""
        if persist_dir and Path(persist_dir).exists():
            from llama_index.core import StorageContext, load_index_from_storage
            
            print(f"Loading index from: {persist_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self.index = load_index_from_storage(storage_context)
        elif documents:
            print("Building new index from documents")
            self.index = VectorStoreIndex.from_documents(
                documents,
                embed_model=self.embed_model
            )
            
            if persist_dir:
                self.index.storage_context.persist(persist_dir=persist_dir)
                print(f"Index saved to: {persist_dir}")
        else:
            raise ValueError("Either documents or persist_dir must be provided")
        
        self.retriever = MultiHopRetriever(
            vector_index=self.index,
            top_k=self.top_k,
            device=self.device,
            tokenizer=self.tokenizer,
            model=self.model
        )
    
    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for Llama3."""
        return f"""<|start_header_id|>user<|end_header_id|>
You are a helpful assistant. Use the following context to answer the question accurately.

Context:
{context}

Question: {query}

Answer the question based on the context above. If the context doesn't contain enough information, say so.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    
    def generate_answer(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate answer using local Llama3 model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.autograd.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split(
                "<|start_header_id|>assistant<|end_header_id|>"
            )[-1].strip()
        
        return response
    
    def query(self, query: str) -> MultiHopRAGResult:
        """
        Execute multi-hop RAG query.
        
        Args:
            query: The user's question
            
        Returns:
            MultiHopRAGResult containing all retrieval and generation details
        """
        total_start = time.perf_counter()
        
        print(f"\n{'='*60}")
        print(f"Processing query: {query}")
        print(f"{'='*60}")
        
        intermediate_results, combined_context = self.retriever.multi_hop_retrieve(query)
        
        total_retrieval_time = sum(r.retrieval_time for r in intermediate_results)
        
        gen_start = time.perf_counter()
        
        prompt = self.format_prompt(query, combined_context)
        final_answer = self.generate_answer(prompt)
        
        total_generation_time = time.perf_counter() - gen_start
        total_time = time.perf_counter() - total_start
        
        print(f"\nRetrieval time: {total_retrieval_time:.3f}s")
        print(f"Generation time: {total_generation_time:.3f}s")
        print(f"Total time: {total_time:.3f}s")
        
        return MultiHopRAGResult(
            original_query=query,
            sub_queries=[r.query for r in intermediate_results],
            intermediate_results=intermediate_results,
            final_answer=final_answer,
            total_retrieval_time=total_retrieval_time,
            total_generation_time=total_generation_time,
            all_retrieved_context=combined_context
        )


class KnowledgeGraphBuilder:
    """
    Builder for knowledge graph enhanced multi-hop retrieval.
    
    This class helps construct entity relationships for better
    multi-hop reasoning.
    """
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.entities = defaultdict(list)
        self.relations = []
    
    def extract_entities(self) -> Dict[str, List[str]]:
        """Extract named entities from documents."""
        entity_pattern = r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        
        for doc in self.documents:
            entities = re.findall(entity_pattern, doc.text)
            self.entities["documents"].extend(entities)
        
        return dict(self.entities)
    
    def build_entity_relations(self) -> List[Tuple[str, str, str]]:
        """
        Build entity-relation triples.
        
        Returns:
            List of (entity, relation, entity) tuples
        """
        relations = []
        
        for doc in self.documents:
            sentences = re.split(r'[.!?]', doc.text)
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                entities = re.findall(
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    sent
                )
                
                if len(entities) >= 2:
                    for i in range(len(entities) - 1):
                        relations.append((
                            entities[i],
                            "related_to",
                            entities[i + 1]
                        ))
        
        self.relations = relations
        return relations
    
    def get_hop_paths(self, source: str, target: str) -> List[List[str]]:
        """
        Find paths between source and target entities.
        
        Used for multi-hop reasoning path discovery.
        """
        graph = defaultdict(list)
        for src, rel, tgt in self.relations:
            graph[src].append((rel, tgt))
        
        paths = []
        
        def dfs(current: str, path: List[str], depth: int):
            if depth > 3:
                return
            if current == target:
                paths.append(path.copy())
                return
            
            for rel, neighbor in graph[current]:
                if neighbor not in path:
                    path.append(f"{rel}:{neighbor}")
                    dfs(neighbor, path, depth + 1)
                    path.pop()
        
        dfs(source, [source], 0)
        return paths


def create_sample_documents() -> List[Document]:
    """Create sample documents for testing multi-hop RAG."""
    sample_texts = [
        "Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity. Einstein worked at the Swiss Patent Office in Bern.",
        
        "The theory of relativity comprises two theories by Albert Einstein: special relativity and general relativity. Special relativity was published in 1905. General relativity was published in 1915.",
        
        "Einstein won the Nobel Prize in Physics in 1921. He was awarded for his services to Theoretical Physics. The photoelectric effect explanation was particularly noted.",
        
        "Marie Curie was a Polish-born physicist and chemist. She conducted pioneering research on radioactivity. Curie won Nobel Prizes in both Physics and Chemistry.",
        
        "The Nobel Prize is awarded annually in several categories. The prize was established by Alfred Nobel's will in 1895. The first prizes were awarded in 1901.",
        
        "Switzerland is a landlocked country in Europe. Bern is the de facto capital of Switzerland. The official capital is actually Bern, not Zurich.",
        
        "Nuclear physics studies atomic nuclei and their constituents. Rutherford discovered the nucleus in 1911. Einstein's E=mc² equation relates mass and energy.",
        
        "Bern is the fifth-largest city in Switzerland. The city has a population of about 140,000. Einstein lived in Bern from 1902 to 1909."
    ]
    
    documents = []
    for i, text in enumerate(sample_texts):
        doc = Document(
            text=text,
            metadata={"source": f"doc_{i}", "chunk": i}
        )
        documents.append(doc)
    
    return documents


def run_evaluation():
    """
    Run evaluation of the multi-hop RAG system.
    
    This function demonstrates the complete workflow:
    1. Setup system with LlamaIndex and Llama3
    2. Create sample knowledge base
    3. Execute multi-hop queries
    4. Evaluate results
    """
    import os
    
    model_path = os.environ.get(
        "LLAMA3_MODEL_PATH",
        "/home/sdu/zhu/models/llama3_8B_instruct/"
    )
    
    embedding_path = os.environ.get(
        "EMBEDDING_MODEL_PATH",
        "/home/sdu/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/"
    )
    
    print("Initializing Multi-hop RAG System...")
    print(f"Model path: {model_path}")
    print(f"Embedding path: {embedding_path}")
    
    documents = create_sample_documents()
    
    rag_system = LlamaIndexMultiHopRAG(
        model_path=model_path,
        embedding_model_path=embedding_path,
        documents=documents,
        persist_dir="./index/multihop_test",
        device="cuda",
        top_k=3
    )
    
    test_queries = [
        "What did Albert Einstein develop and when was it published?",
    ]
    
    results = []
    
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)
    
    for query in test_queries:
        result = rag_system.query(query)
        results.append(result)
        
        print(f"\n{'='*60}")
        print(f"Query: {result.original_query}")
        print(f"{'='*60}")
        print(f"Sub-queries: {result.sub_queries}")
        print(f"\nFinal Answer:\n{result.final_answer}")
        print(f"\nRetrieval Time: {result.total_retrieval_time:.3f}s")
        print(f"Generation Time: {result.total_generation_time:.3f}s")
        print(f"Total Time: {result.total_retrieval_time + result.total_generation_time:.3f}s")
    
    summary = {
        "num_queries": len(results),
        "avg_retrieval_time": np.mean([r.total_retrieval_time for r in results]),
        "avg_generation_time": np.mean([r.total_generation_time for r in results]),
        "total_time": sum(r.total_retrieval_time + r.total_generation_time for r in results)
    }
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total queries: {summary['num_queries']}")
    print(f"Avg retrieval time: {summary['avg_retrieval_time']:.3f}s")
    print(f"Avg generation time: {summary['avg_generation_time']:.3f}s")
    print(f"Total time: {summary['total_time']:.3f}s")
    
    return results, summary


if __name__ == "__main__":
    results, summary = run_evaluation()
    
    output_file = "./results/multihop_rag_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "summary": summary,
            "results": [
                {
                    "query": r.original_query,
                    "sub_queries": r.sub_queries,
                    "final_answer": r.final_answer,
                    "retrieval_time": r.total_retrieval_time,
                    "generation_time": r.total_generation_time
                }
                for r in results
            ]
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
