#!/usr/bin/env python3
"""Mathematics Textbook Data Processing Pipeline.

Processes math textbook JSON files (QA pairs and teaching scripts),
stores them in vector databases, and supports retrieval testing.

Input files (in --data-root directory):
  - generate_qa.json: Question-Answer pairs
  - teaching_script_generate.json: Interactive teaching scripts

Usage:
    # Full processing
    PYTHONPATH=. python scripts/test_math_textbook_e2e.py \\
        --data-root /path/to/math_data \\
        --kb-id kb_math_001

    # With clean
    PYTHONPATH=. python scripts/test_math_textbook_e2e.py \\
        --data-root /path/to/math_data \\
        --kb-id kb_math_001 \\
        --clean

    # Query only
    PYTHONPATH=. python scripts/test_math_textbook_e2e.py \\
        --kb-id kb_math_001 \\
        --query-only

    # View sample data
    PYTHONPATH=. python scripts/test_math_textbook_e2e.py \\
        --kb-id kb_math_001 \\
        --sample-only
"""
import argparse
import asyncio
import hashlib
import json
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db
from app.core.database import get_database, mongodb
from app.services.rag_service import RAGRetrieval
from app.utils.embeddings import get_embedding
from app.core.logging import get_logger
from app.models.database import KnowledgeBaseModel

logger = get_logger(__name__)

# Field name constants
KB_ID = "kb_id"
DOC_ID = "doc_id"
CONTENT_TYPE = "content_type"
BOOK_TITLE = "book_title"
CHAPTER_TITLE = "chapter_title"
SECTION_TITLE = "section_title"
CHUNK_INDEX = "chunk_index"
CONTEXT_TEXT = "context_text"

# Math-specific test queries
MATH_TEST_QUERIES = [
    {"type": "简单概念", "query": "什么是集合的表示方法", "description": "基础概念召回"},
    {"type": "方法步骤", "query": "如何求函数的定义域", "description": "解题步骤召回"},
    {"type": "例题解析", "query": "集合交并集的运算例题", "description": "例题相关内容召回"},
    {"type": "对比辨析", "query": "函数的单调性和奇偶性有什么区别", "description": "概念对比召回"},
    {"type": "常见错误", "query": "求函数定义域时常见的错误有哪些", "description": "易错点召回"},
    {"type": "应用拓展", "query": "集合在实际生活中的应用", "description": "拓展应用召回"},
    {"type": "简单概念", "query": "什么是集合", "description": "基础概念召回"},
]


@dataclass
class MathChunk:
    """Math textbook chunk data model."""

    chunk_id: str
    content: str
    doc_id: str
    kb_id: str
    chunk_index: int
    content_type: str  # "qa" or "teaching_script"
    book_title: str
    chapter_title: str
    section_title: str

    # Separation of index and context: questions for index/embedding, answers for LLM context
    embedding_text: str = ""  # For ES/ChromaDB index and embedding (questions)
    context_text: str = ""  # For LLM context (answers)

    # Optional fields for original data
    question: Optional[str] = None
    answer: Optional[str] = None
    student_question: Optional[str] = None
    teaching_script: Optional[str] = None

    # New fields for updated teaching_script_generate.json schema
    natural_questions: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    teaching_script_tts: Optional[str] = None


# ============================================================================
# Database Utility Functions
# ============================================================================

async def _connect_databases():
    """Connect to all databases and print connection status."""
    # MongoDB and ElasticSearch are async, ChromaDB is sync
    await mongodb.connect()
    print(f"  MongoDB connected")

    chroma_db.connect()
    print(f"  ChromaDB connected")

    await es_db.connect()
    print(f"  ElasticSearch connected")


async def _disconnect_databases():
    """Disconnect from MongoDB."""
    try:
        await mongodb.disconnect()
    except Exception:
        pass


async def _clear_data_by_filter(filter_dict: Dict[str, str], filter_name: str):
    """Clear data from all databases using a filter dictionary.

    Args:
        filter_dict: Dictionary with KB_ID or DOC_ID as key
        filter_name: Description of the filter for logging
    """
    print(f"  Clearing data by {filter_name}...")

    db = await get_database()

    # MongoDB
    doc_result = await db.documents.delete_many(filter_dict)
    chunk_result = await db.document_chunks.delete_many(filter_dict)
    print(f"  MongoDB: deleted {doc_result.deleted_count} docs, {chunk_result.deleted_count} chunks")

    # ChromaDB
    if chroma_db.client:
        try:
            chroma_db.doc_collection.delete(where=filter_dict)
            print(f"  ChromaDB: deleted data for {filter_name}")
        except Exception as e:
            print(f"  ChromaDB delete: {e}")

    # ElasticSearch
    try:
        await es_db.delete_by_query(
            index=es_db.doc_index,
            body={"query": {"term": {list(filter_dict.keys())[0]: list(filter_dict.values())[0]}}},
        )
        print(f"  ElasticSearch: deleted data for {filter_name}")
    except Exception as e:
        print(f"  ElasticSearch delete: {e}")


# ============================================================================
# Math Textbook Processor
# ============================================================================

class MathTextbookProcessor:
    """Mathematics textbook data processor."""

    def __init__(
        self,
        data_root: str,
        kb_id: str,
        chunk_size: int = 4096,
        enable_rerank: bool = True,
        top_k: int = 10,
        max_chunks: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.kb_id = kb_id
        self.kb_name = self.data_root.name
        self.chunk_size = chunk_size
        self.enable_rerank = enable_rerank
        self.top_k = top_k
        self.max_chunks = max_chunks

        # Generate fixed document ID based on folder name and kb_id
        self.doc_id = self._generate_doc_id(self.data_root.name, self.kb_id)

        # File paths
        self.qa_json_path = self.data_root / "generate_qa.json"
        self.script_json_path = self.data_root / "teaching_script_generate.json"

        # Storage
        self.chunks: List[MathChunk] = []
        self.results: Dict[str, Any] = {
            "data_root": str(self.data_root),
            KB_ID: self.kb_id,
            "kb_name": self.kb_name,
            DOC_ID: self.doc_id,
            "timestamp": datetime.now().isoformat(),
        }

    async def setup(self):
        """Setup test environment."""
        print("\n" + "=" * 80)
        print("Mathematics Textbook Data Processing Pipeline")
        print("=" * 80)

        print(f"\nConfiguration:")
        print(f"  Data Root: {self.data_root}")
        print(f"  KB ID: {self.kb_id}")
        print(f"  KB Name: {self.kb_name}")
        print(f"  Doc ID: {self.doc_id}")
        print(f"  Chunk Size: {self.chunk_size}")
        print(f"  Rerank: {'Enabled' if self.enable_rerank else 'Disabled'}")
        print(f"  Max Chunks: {self.max_chunks if self.max_chunks else 'unlimited'}")

        # Verify data root exists
        if not self.data_root.exists():
            print(f"\nError: Data root directory not found: {self.data_root}")
            raise FileNotFoundError(f"Directory not found: {self.data_root}")

        # Connect to databases
        print(f"\nConnecting to databases...")
        await _connect_databases()

        # Ensure knowledge base exists
        await self._ensure_kb_exists()

    async def _ensure_kb_exists(self):
        """Ensure knowledge base exists, create if not."""
        print(f"\nChecking knowledge base...")

        try:
            db = await get_database()
            kb = await db.knowledge_bases.find_one({KB_ID: self.kb_id})

            if not kb:
                kb_model = KnowledgeBaseModel(
                    kb_id=self.kb_id,
                    name=self.kb_name,
                    description=f"{self.kb_name} - 数学教材知识库",
                    status="active",
                )
                await db.knowledge_bases.insert_one(kb_model.model_dump())
                print(f"  Created knowledge base: {self.kb_name} ({self.kb_id})")
            else:
                existing_name = kb.get("name", "")
                print(f"  Using existing knowledge base: {existing_name} ({self.kb_id})")
        except Exception as e:
            print(f"  Knowledge base check failed: {e}")

    def _parse_json_file(self, file_path: Path, file_type: str) -> tuple[Dict, Dict]:
        """Parse a JSON file and return (data, stats).

        Args:
            file_path: Path to the JSON file
            file_type: Type of file ("qa" or "teaching_script")

        Returns:
            Tuple of (parsed_data, statistics_dict)
        """
        if not file_path.exists():
            return {}, {}

        data = json.loads(file_path.read_text(encoding="utf-8"))
        stats = self._extract_stats(data, file_type)
        return data, stats

    def _extract_stats(self, data: Dict, content_type: str) -> Dict[str, int]:
        """Extract statistics from parsed data."""
        chapters = data.get("chapters", [])
        chapter_count = len(chapters)
        section_count = 0
        item_count = 0

        for chapter in chapters:
            sections = chapter.get("sections", [])
            section_count += len(sections)

            for section in sections:
                if content_type == "qa":
                    item_count += len(section.get("qa_pairs", []))
                else:  # teaching_script
                    script = section.get("teaching_script", "")
                    if script and script.strip():
                        item_count += 1

        return {
            "chapters": chapter_count,
            "sections": section_count,
            "items": item_count,
        }

    async def step1_parse_json(self) -> bool:
        """Step 1: Parse JSON files."""
        print("\n" + "-" * 80)
        print("Step 1: Parsing JSON Files")
        print("-" * 80)

        try:
            # qa_data, qa_stats = self._parse_json_file(self.qa_json_path, "qa")
            # if qa_data:
            #     print(f"\n  QA File: {self.qa_json_path.name}")
            #     print(f"    Book Title: {qa_data.get('book_title', 'N/A')}")
            #     print(f"    Chapters: {qa_stats.get('chapters', 0)}")
            #     print(f"    Sections: {qa_stats.get('sections', 0)}")
            #     print(f"    QA Pairs: {qa_stats.get('items', 0)}")
            # else:
            #     print(f"\n  QA File not found: {self.qa_json_path.name}")

            script_data, script_stats = self._parse_json_file(self.script_json_path, "teaching_script")
            if script_data:
                print(f"\n  Teaching Script File: {self.script_json_path.name}")
                print(f"    Book Title: {script_data.get('book_title', 'N/A')}")
                print(f"    Chapters: {script_stats.get('chapters', 0)}")
                print(f"    Sections: {script_stats.get('sections', 0)}")
                print(f"    Teaching Scripts: {script_stats.get('items', 0)}")
            else:
                print(f"\n  Teaching Script File not found: {self.script_json_path.name}")

            self.results["parse_result"] = {
                # "qa": qa_stats,
                "teaching_script": script_stats,
            }

            return True

        except Exception as e:
            print(f"  Parse failed: {e}")
            traceback.print_exc()
            return False

    def _create_chunk(
        self,
        chunk_id: str,
        content: str,
        chunk_idx: int,
        content_type: str,
        book_title: str,
        chapter_title: str,
        section_title: str,
        embedding_text: str,
        context_text: str,
        **extra_fields,
    ) -> MathChunk:
        """Create a MathChunk with common fields."""
        fields = {
            "chunk_id": chunk_id,
            "content": content,
            "doc_id": self.doc_id,
            KB_ID: self.kb_id,
            "chunk_index": chunk_idx,
            CONTENT_TYPE: content_type,
            BOOK_TITLE: book_title,
            CHAPTER_TITLE: chapter_title,
            SECTION_TITLE: section_title,
            "embedding_text": embedding_text,
            "context_text": context_text,
        }
        fields.update(extra_fields)
        return MathChunk(**fields)

    def _extract_chunks_from_qa(self, qa_data: Dict) -> List[MathChunk]:
        """Extract chunks from QA data."""
        chunks = []
        book_title = qa_data.get("book_title", "")
        chunk_idx = 0

        for chapter in qa_data.get("chapters", []):
            chapter_title = chapter.get("chapter_title", "")

            for section in chapter.get("sections", []):
                section_title = section.get("section_title", "")

                for qa_pair in section.get("qa_pairs", []):
                    question = qa_pair.get("question", "")
                    answer = qa_pair.get("answer", "")

                    if not question or not answer:
                        continue

                    content = f"Q: {question}\n\nA: {answer}"
                    chunks.append(self._create_chunk(
                        chunk_id=f"{self.doc_id}_qa_{chunk_idx}",
                        content=content,
                        chunk_idx=chunk_idx,
                        content_type="qa",
                        book_title=book_title,
                        chapter_title=chapter_title,
                        section_title=section_title,
                        embedding_text=question,
                        context_text=answer,
                        question=question,
                        answer=answer,
                    ))
                    chunk_idx += 1

        return chunks

    def _extract_chunks_from_teaching_script(self, script_data: Dict) -> List[MathChunk]:
        """Extract chunks from teaching script data.

        New schema: each section has:
        - teaching_script: complete teacher explanation
        - teaching_script_tts: TTS-optimized version
        - knowledge_summary.natural_questions: list of 2-4 natural questions
        - knowledge_summary.keywords: list of 3-6 keywords

        Each natural_question creates a separate chunk for better retrieval precision.
        All chunks from the same section share the same teaching_script content.
        """
        chunks = []
        book_title = script_data.get("book_title", "")
        chunk_idx = 0

        for chapter in script_data.get("chapters", []):
            chapter_title = chapter.get("chapter_title", "")

            for section in chapter.get("sections", []):
                section_title = section.get("section_title", "")
                teaching_script = section.get("teaching_script", "")
                teaching_script_tts = section.get("teaching_script_tts", "")
                knowledge_summary = section.get("knowledge_summary", {})

                if not teaching_script or not teaching_script.strip():
                    continue

                natural_questions = knowledge_summary.get("natural_questions", [])
                keywords = knowledge_summary.get("keywords", [])

                # If no natural_questions, create one chunk with section title
                if not natural_questions:
                    chunks.append(self._create_chunk(
                        chunk_id=f"{self.doc_id}_script_{chunk_idx}",
                        content=teaching_script,
                        chunk_idx=chunk_idx,
                        content_type="teaching_script",
                        book_title=book_title,
                        chapter_title=chapter_title,
                        section_title=section_title,
                        embedding_text=section_title,
                        context_text=teaching_script,
                        teaching_script=teaching_script,
                        teaching_script_tts=teaching_script_tts,
                        natural_questions=natural_questions,
                        keywords=keywords,
                    ))
                    chunk_idx += 1
                else:
                    # Create one chunk per natural_question for better retrieval
                    for natural_question in natural_questions:
                        chunks.append(self._create_chunk(
                            chunk_id=f"{self.doc_id}_script_{chunk_idx}",
                            content=teaching_script,
                            chunk_idx=chunk_idx,
                            content_type="teaching_script",
                            book_title=book_title,
                            chapter_title=chapter_title,
                            section_title=section_title,
                            embedding_text=natural_question,  # Single question for indexing
                            context_text=teaching_script,  # Full script for LLM context
                            teaching_script=teaching_script,
                            teaching_script_tts=teaching_script_tts,
                            natural_questions=natural_questions,  # All questions stored
                            keywords=keywords,
                        ))
                        chunk_idx += 1

        return chunks

    async def step2_clear_old_data(self, force: bool = False) -> bool:
        """Step 2: Clear old data for current doc_id (optional)."""
        print("\n" + "-" * 80)
        print("Step 2: Clearing Old Data")
        print("-" * 80)

        if not force:
            print("  Skipped (use --clean to enable)")
            return True

        await _clear_data_by_filter({DOC_ID: self.doc_id}, f"doc_id={self.doc_id}")
        return True

    async def step3_chunk_and_vectorize(self) -> bool:
        """Step 3: Chunk and vectorize."""
        print("\n" + "-" * 80)
        print("Step 3: Chunking and Vectorizing")
        print("-" * 80)

        try:
            self.chunks = []

            # Extract chunks from QA
            if self.qa_json_path.exists():
                qa_data = json.loads(self.qa_json_path.read_text(encoding="utf-8"))
                qa_chunks = self._extract_chunks_from_qa(qa_data)
                self.chunks.extend(qa_chunks)
                print(f"  QA chunks: {len(qa_chunks)}")

            # Extract chunks from teaching scripts
            if self.script_json_path.exists():
                script_data = json.loads(self.script_json_path.read_text(encoding="utf-8"))
                script_chunks = self._extract_chunks_from_teaching_script(script_data)
                self.chunks.extend(script_chunks)
                print(f"  Teaching script chunks: {len(script_chunks)}")

            if not self.chunks:
                print(f"  No chunks extracted!")
                return False

            # Apply max_chunks limit
            if self.max_chunks and len(self.chunks) > self.max_chunks:
                original_count = len(self.chunks)
                self.chunks = self.chunks[:self.max_chunks]
                print(f"  Limited to {self.max_chunks} chunks (skipped {original_count - self.max_chunks})")

            print(f"  Total chunks: {len(self.chunks)}")

            # Get embedder and test
            embedder = get_embedding()
            print(f"  Embedder: {type(embedder).__name__}")
            test_embedding = embedder.embed_query("测试文本")
            print(f"  Embedding dimension: {len(test_embedding)}")

            # Statistics
            total_chars = sum(len(c.content) for c in self.chunks)
            avg_chars = total_chars // len(self.chunks)
            max_chars = max(len(c.content) for c in self.chunks)
            min_chars = min(len(c.content) for c in self.chunks)

            print(f"  Total characters: {total_chars}")
            print(f"  Average chunk size: {avg_chars} chars")
            print(f"  Max chunk size: {max_chars} chars")
            print(f"  Min chunk size: {min_chars} chars")

            # Count by content type
            qa_count = sum(1 for c in self.chunks if c.content_type == "qa")
            script_count = sum(1 for c in self.chunks if c.content_type == "teaching_script")
            print(f"  QA chunks: {qa_count}")
            print(f"  Teaching script chunks: {script_count}")

            self.results["chunks"] = {
                "total": len(self.chunks),
                "qa_count": qa_count,
                "script_count": script_count,
                "total_chars": total_chars,
                "avg_chars": avg_chars,
            }

            return True

        except Exception as e:
            print(f"  Chunking failed: {e}")
            traceback.print_exc()
            return False

    async def step4_store_databases(self) -> bool:
        """Step 4: Store to databases."""
        print("\n" + "-" * 80)
        print("Step 4: Storing to Databases")
        print("-" * 80)

        try:
            if not self.chunks:
                print(f"  No chunks to store!")
                return False

            # Prepare data using embedding_text for index/embedding
            chunk_texts = [c.embedding_text for c in self.chunks]
            chunk_ids = [c.chunk_id for c in self.chunks]
            chunk_metadatas = [
                {
                    DOC_ID: c.doc_id,
                    KB_ID: c.kb_id,
                    CHUNK_INDEX: c.chunk_index,
                    CONTENT_TYPE: c.content_type,
                    BOOK_TITLE: c.book_title,
                    CHAPTER_TITLE: c.chapter_title,
                    SECTION_TITLE: c.section_title,
                    CONTEXT_TEXT: c.context_text,
                }
                for c in self.chunks
            ]

            # Store in ChromaDB
            if chroma_db.client:
                await chroma_db.add_documents(
                    collection_name="doc",
                    documents=chunk_texts,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids,
                )
                print(f"  ChromaDB: stored {len(chunk_ids)} chunks")

            # Store in ElasticSearch
            for chunk in self.chunks:
                es_document = {
                    "chunk_id": chunk.chunk_id,
                    DOC_ID: chunk.doc_id,
                    KB_ID: chunk.kb_id,
                    "content": chunk.embedding_text,
                    CONTEXT_TEXT: chunk.context_text,
                    CHUNK_INDEX: chunk.chunk_index,
                    CONTENT_TYPE: chunk.content_type,
                    BOOK_TITLE: chunk.book_title,
                    CHAPTER_TITLE: chunk.chapter_title,
                    SECTION_TITLE: chunk.section_title,
                }
                await es_db.index_document(
                    index=es_db.doc_index,
                    doc_id=chunk.chunk_id,
                    document=es_document,
                )
            print(f"  ElasticSearch: stored {len(self.chunks)} chunks")

            # Store in MongoDB (full data)
            db = await get_database()
            chunk_docs = [
                {
                    "chunk_id": c.chunk_id,
                    DOC_ID: c.doc_id,
                    KB_ID: c.kb_id,
                    "content": c.content,
                    "embedding_text": c.embedding_text,
                    "context_text": c.context_text,
                    CHUNK_INDEX: c.chunk_index,
                    CONTENT_TYPE: c.content_type,
                    BOOK_TITLE: c.book_title,
                    CHAPTER_TITLE: c.chapter_title,
                    SECTION_TITLE: c.section_title,
                    "question": c.question,
                    "answer": c.answer,
                    "student_question": c.student_question,
                    "teaching_script": c.teaching_script,
                    # New fields for updated schema
                    "natural_questions": c.natural_questions,
                    "keywords": c.keywords,
                    "teaching_script_tts": c.teaching_script_tts,
                }
                for c in self.chunks
            ]

            await db.document_chunks.insert_many(chunk_docs)
            print(f"  MongoDB: stored {len(chunk_docs)} chunks")

            return True

        except Exception as e:
            print(f"  Storage failed: {e}")
            traceback.print_exc()
            return False

    async def step5_retrieve_and_score(self) -> bool:
        """Step 5: Retrieve and score."""
        print("\n" + "-" * 80)
        print("Step 5: Retrieval and Scoring")
        print("-" * 80)

        try:
            retrieval_results = []

            for test_case in MATH_TEST_QUERIES:
                print(f"\n  Query Type: {test_case['type']}")
                print(f"     Query: {test_case['query']}")

                rag = RAGRetrieval()
                results = await rag.search(
                    query=test_case["query"],
                    kb_ids=[self.kb_id],
                    top_k=self.top_k,
                    enable_rerank=self.enable_rerank,
                )

                print(f"     Retrieved: {len(results)} results")

                # Show top results
                for i, doc in enumerate(results[:3]):
                    self._print_result(i, doc)

                retrieval_results.append({
                    "type": test_case["type"],
                    "query": test_case["query"],
                    "results_count": len(results),
                    "top_scores": [r.get("rrf_score", r.get("score", 0)) for r in results[:3]],
                })

            self.results["retrieval"] = retrieval_results
            return True

        except Exception as e:
            print(f"  Retrieval failed: {e}")
            traceback.print_exc()
            return False

    def _print_result(self, index: int, doc: Dict):
        """Print a retrieval result."""
        content_preview = doc.get("content", "")[:100]
        score = doc.get("rrf_score", doc.get("score", 0))
        content_type = doc.get(CONTENT_TYPE, "unknown")
        chapter = doc.get(CHAPTER_TITLE, "")[:30]
        section = doc.get(SECTION_TITLE, "")[:30]

        print(f"       [{index + 1}] Score: {score:.4f} | Type: {content_type}")
        if chapter:
            print(f"           Chapter: {chapter}")
        if section:
            print(f"           Section: {section}")
        print(f"           Content: {content_preview}...")

    async def show_sample_data(self):
        """Show sample data from databases."""
        print("\n" + "-" * 80)
        print("Sample Data")
        print("-" * 80)

        # MongoDB stats
        await self._show_mongodb_samples()

        # ChromaDB stats
        await self._show_chromadb_samples()

        # ElasticSearch stats
        await self._show_elasticsearch_samples()

    async def _show_mongodb_samples(self):
        """Show MongoDB sample data."""
        try:
            db = await get_database()
            chunk_count = await db.document_chunks.count_documents({KB_ID: self.kb_id})
            print(f"\n  MongoDB chunks for {KB_ID}={self.kb_id}: {chunk_count}")

            if chunk_count > 0:
                sample_chunks = await db.document_chunks.find({KB_ID: self.kb_id}).limit(3).to_list(None)
                print(f"\n  Sample chunks (first 3):")
                for i, chunk in enumerate(sample_chunks):
                    content_preview = chunk.get("content", "")[:150]
                    content_type = chunk.get(CONTENT_TYPE, "unknown")
                    chapter = chunk.get(CHAPTER_TITLE, "")[:40]
                    section = chunk.get(SECTION_TITLE, "")[:40]
                    print(f"    [{i + 1}] chunk_id: {chunk.get('chunk_id')}")
                    print(f"        content_type: {content_type}")
                    print(f"        chapter: {chapter}")
                    print(f"        section: {section}")
                    print(f"        content: {content_preview}...")
                    print()
        except Exception as e:
            print(f"  MongoDB query failed: {e}")

    async def _show_chromadb_samples(self):
        """Show ChromaDB sample data."""
        try:
            if not chroma_db.client or not chroma_db.doc_collection:
                return

            results = chroma_db.doc_collection.get(
                where={KB_ID: self.kb_id},
                limit=3
            )
            print(f"  ChromaDB chunks for {KB_ID}={self.kb_id}: {len(results.get('ids', []))}")

            if results.get("ids"):
                print(f"\n  Sample chunks (first 3):")
                for i, chunk_id in enumerate(results["ids"][:3]):
                    doc = results["documents"][i] if i < len(results["documents"]) else ""
                    metadata = results["metadatas"][i] if i < len(results["metadatas"]) else {}
                    print(f"    [{i + 1}] id: {chunk_id}")
                    print(f"        content_type: {metadata.get(CONTENT_TYPE, 'unknown')}")
                    print(f"        content: {doc[:150]}...")
                    print()
        except Exception as e:
            print(f"  ChromaDB query failed: {e}")

    async def _show_elasticsearch_samples(self):
        """Show ElasticSearch sample data."""
        try:
            count_result = await es_db.client.count(
                index=es_db.doc_index,
                body={"query": {"term": {KB_ID: self.kb_id}}}
            )
            print(f"  ElasticSearch documents for {KB_ID}={self.kb_id}: {count_result['count']}")

            if count_result["count"] > 0:
                search_result = await es_db.client.search(
                    index=es_db.doc_index,
                    body={"query": {"term": {KB_ID: self.kb_id}}, "size": 3}
                )

                if search_result["hits"]["hits"]:
                    print(f"\n  Sample documents (first 3):")
                    for i, hit in enumerate(search_result["hits"]["hits"]):
                        source = hit["_source"]
                        content = source.get("content", "")[:150]
                        content_type = source.get(CONTENT_TYPE, "unknown")
                        print(f"    [{i + 1}] id: {hit['_id']}")
                        print(f"        content_type: {content_type}")
                        print(f"        content: {content}...")
                        print()
        except Exception as e:
            print(f"  ElasticSearch query failed: {e}")

    async def run(self, clean: bool = False) -> bool:
        """Run complete pipeline."""
        await self.setup()

        steps = [
            ("Step 1: Parse JSON", self.step1_parse_json),
            ("Step 2: Clear old data", lambda: self.step2_clear_old_data(force=clean)),
            ("Step 3: Chunk and vectorize", self.step3_chunk_and_vectorize),
            ("Step 4: Store to databases", self.step4_store_databases),
            ("Step 5: Retrieve and score", self.step5_retrieve_and_score),
        ]

        for step_name, step_func in steps:
            try:
                if not await step_func():
                    await _disconnect_databases()
                    return False
            except Exception as e:
                print(f"\n{step_name} failed: {e}")
                traceback.print_exc()
                await _disconnect_databases()
                return False

        # Print summary
        self._print_summary()
        await _disconnect_databases()
        return True

    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"  KB ID: {self.kb_id}")
        print(f"  KB Name: {self.kb_name}")
        print(f"  Doc ID: {self.doc_id}")
        print(f"  Total chunks: {len(self.chunks)}")
        print(f"  QA chunks: {sum(1 for c in self.chunks if c.content_type == 'qa')}")
        print(f"  Teaching script chunks: {sum(1 for c in self.chunks if c.content_type == 'teaching_script')}")

    @staticmethod
    def _generate_doc_id(folder_name: str, kb_id: str) -> str:
        """Generate fixed document ID based on folder name and kb_id.

        Same folder_name + kb_id always produces the same doc_id.
        """
        content = f"{folder_name}_{kb_id}"
        hash_obj = hashlib.md5(content.encode())
        return f"doc_{hash_obj.hexdigest()[:12]}"


# ============================================================================
# CLI and Main Entry Point
# ============================================================================

def _print_test_query_results(results: List[Dict]):
    """Print full query results for query-only mode."""
    for i, doc in enumerate(results):
        content_preview = doc.get("content", "")[:150]
        score = doc.get("rrf_score", doc.get("score", 0))
        content_type = doc.get(CONTENT_TYPE, "unknown")
        chapter = doc.get(CHAPTER_TITLE, "")[:30]
        section = doc.get(SECTION_TITLE, "")[:30]

        print(f"  [{i + 1}] Score: {score:.4f} | Type: {content_type}")
        if chapter:
            print(f"      Chapter: {chapter}")
        if section:
            print(f"      Section: {section}")
        print(f"      Content: {content_preview}...")
        print()


async def _run_clean_all_mode(kb_id: str):
    """Run clean-all mode to clear entire knowledge base."""
    print("\n" + "=" * 80)
    print("Clean All Mode (Entire Knowledge Base)")
    print("=" * 80)
    print(f"KB ID: {kb_id}")

    await _connect_databases()

    print("\n" + "-" * 80)
    print("Cleaning All Data (by kb_id)")
    print("-" * 80)

    await _clear_data_by_filter({KB_ID: kb_id}, f"{KB_ID}={kb_id}")

    print("\nClean all completed!")
    await _disconnect_databases()


async def _run_clean_only_mode(data_root: str, kb_id: str):
    """Run clean-only mode to clear current textbook."""
    print("\n" + "=" * 80)
    print("Clean Only Mode (Current Textbook)")
    print("=" * 80)
    print(f"Data Root: {data_root}")
    print(f"KB ID: {kb_id}")

    doc_id = MathTextbookProcessor._generate_doc_id(Path(data_root).name, kb_id)
    print(f"Doc ID: {doc_id}")

    await _connect_databases()

    print("\n" + "-" * 80)
    print("Cleaning Data (by doc_id)")
    print("-" * 80)

    await _clear_data_by_filter({DOC_ID: doc_id}, f"{DOC_ID}={doc_id}")

    print("\nClean completed!")
    await _disconnect_databases()


async def _run_query_only_mode(kb_id: str, no_rerank: bool, top_k: int, custom_query: Optional[str] = None):
    """Run query-only mode.

    Args:
        kb_id: Knowledge base ID
        no_rerank: Whether to disable reranking
        top_k: Number of results to retrieve
        custom_query: Optional custom query text. If provided, only run this query.
    """
    print("\n" + "=" * 80)
    print("Query Only Mode")
    print("=" * 80)
    print(f"KB ID: {kb_id}")
    print(f"Rerank: {'Disabled' if no_rerank else 'Enabled'}")

    await _connect_databases()

    # Use custom query or predefined test queries
    if custom_query:
        print(f"\nRunning custom query...\n")
        queries_to_run = [{"type": "Custom", "query": custom_query}]
    else:
        print(f"\nRunning {len(MATH_TEST_QUERIES)} predefined queries...\n")
        queries_to_run = MATH_TEST_QUERIES

    for test_case in queries_to_run:
        print("-" * 80)
        print(f"Query Type: {test_case['type']}")
        print(f"Query: {test_case['query']}")

        rag = RAGRetrieval()
        results = await rag.search(
            query=test_case["query"],
            kb_ids=[kb_id],
            top_k=top_k,
            enable_rerank=not no_rerank,
        )

        print(f"Retrieved: {len(results)} results\n")
        _print_test_query_results(results)

    await _disconnect_databases()


async def _run_sample_only_mode(kb_id: str, max_chunks: Optional[int]):
    """Run sample-only mode."""
    print("\n" + "=" * 80)
    print("Sample Data Mode")
    print("=" * 80)
    print(f"KB ID: {kb_id}")

    await _connect_databases()

    processor = MathTextbookProcessor(
        data_root="/tmp",  # dummy value
        kb_id=kb_id,
        max_chunks=max_chunks,
    )
    await processor.show_sample_data()

    await _disconnect_databases()


async def _run_batch_mode(
    batch_dirs: List[str],
    kb_id: str,
    chunk_size: int,
    no_rerank: bool,
    top_k: int,
    max_chunks: Optional[int],
    clean: bool,
):
    """Run batch processing mode."""
    print("\n" + "=" * 80)
    print("Batch Processing Mode")
    print("=" * 80)
    print(f"KB ID: {kb_id}")
    print(f"Directories: {len(batch_dirs)}")

    # Validate directories
    valid_dirs = [d for d in batch_dirs if Path(d).exists()]
    for d in batch_dirs:
        if d not in valid_dirs:
            print(f"  Warning: Directory not found, skipping: {d}")

    if not valid_dirs:
        print(f"\nError: No valid directories found", file=sys.stderr)
        sys.exit(1)

    print(f"  Valid directories: {len(valid_dirs)}")

    # Process each directory
    success_count = 0
    fail_count = 0

    for i, data_root in enumerate(valid_dirs, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(valid_dirs)}] Processing: {data_root}")
        print(f"{'=' * 80}")

        processor = MathTextbookProcessor(
            data_root=data_root,
            kb_id=kb_id,
            chunk_size=chunk_size,
            enable_rerank=not no_rerank,
            top_k=top_k,
            max_chunks=max_chunks,
        )

        try:
            success = await processor.run(clean=clean)
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"\nFailed: {e}")
            traceback.print_exc()
            fail_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("Batch Processing Summary")
    print("=" * 80)
    print(f"  Total: {len(valid_dirs)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    sys.exit(0 if fail_count == 0 else 1)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mathematics Textbook Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full processing (single textbook)
  %(prog)s --data-root /path/to/math_data --kb-id kb_math_textbook

  # With clean (clear current textbook before import)
  %(prog)s --data-root /path/to/math_data --kb-id kb_math_textbook --clean

  # Batch processing multiple textbooks
  %(prog)s --batch /path/to/book1 /path/to/book2 --kb-id kb_math_textbook

  # Clean entire knowledge base (all textbooks)
  %(prog)s --kb-id kb_math_textbook --clean-all

  # Query only (predefined test queries)
  %(prog)s --query-only --kb-id kb_math_textbook

  # Query only (custom query)
  %(prog)s --query-only --kb-id kb_math_textbook --query "什么是向量的点积？"

  # Query with custom options
  %(prog)s --query-only --kb-id kb_math_textbook --query "如何求函数定义域" --top-k 5 --no-rerank

  # Sample only
  %(prog)s --sample-only --kb-id kb_math_textbook
        """
    )

    parser.add_argument("--data-root", help="JSON files directory (required for single processing)")
    parser.add_argument("--kb-id", required=True, help="Knowledge base ID")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Max chunk size in characters (default: 4096)")
    parser.add_argument("--batch", nargs="+", help="Batch process multiple directories")
    parser.add_argument("--clean", action="store_true", help="Clear current textbook data before processing (by doc_id)")
    parser.add_argument("--clean-all", action="store_true", help="Clear entire knowledge base data (by kb_id)")
    parser.add_argument("--clean-only", action="store_true", help="Only clean current textbook data, then exit (by doc_id)")
    parser.add_argument("--query-only", action="store_true", help="Only run retrieval queries")
    parser.add_argument("--query", help="Custom query text (use with --query-only)")
    parser.add_argument("--sample-only", action="store_true", help="Show sample data from databases")
    parser.add_argument("--no-rerank", action="store_true", help="Disable BGE Reranker")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve (default: 10)")
    parser.add_argument("--max-chunks", type=int, help="Maximum chunks to import (for testing)")

    args = parser.parse_args()

    # Route to appropriate mode
    try:
        if args.clean_all:
            await _run_clean_all_mode(args.kb_id)
        elif args.clean_only:
            if not args.data_root:
                print(f"Error: --data-root is required for --clean-only (to determine doc_id)", file=sys.stderr)
                sys.exit(1)
            await _run_clean_only_mode(args.data_root, args.kb_id)
        elif args.query_only:
            await _run_query_only_mode(args.kb_id, args.no_rerank, args.top_k, args.query)
        elif args.sample_only:
            await _run_sample_only_mode(args.kb_id, args.max_chunks)
        elif args.batch:
            await _run_batch_mode(
                args.batch, args.kb_id, args.chunk_size,
                args.no_rerank, args.top_k, args.max_chunks, args.clean,
            )
        else:
            # Full processing mode (single directory)
            if not args.data_root:
                print(f"Error: --data-root is required for single processing mode", file=sys.stderr)
                sys.exit(1)

            data_root = Path(args.data_root)
            if not data_root.exists():
                print(f"Error: Directory not found: {args.data_root}", file=sys.stderr)
                sys.exit(1)

            processor = MathTextbookProcessor(
                data_root=str(data_root),
                kb_id=args.kb_id,
                chunk_size=args.chunk_size,
                enable_rerank=not args.no_rerank,
                top_k=args.top_k,
                max_chunks=args.max_chunks,
            )

            success = await processor.run(clean=args.clean)
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nFailed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
