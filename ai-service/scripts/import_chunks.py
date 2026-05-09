#!/usr/bin/env python3
"""
通用 Chunk 数据导入脚本

支持导入 textbook-chunker 格式的 chunks.json 到三数据库 (MongoDB + ChromaDB + ElasticSearch)
同时保留完整的测试功能（导入 + 检索测试 + 评分）

Usage:
    # 完整导入流程
    python scripts/import_chunks.py --chunks chunks.json --kb-id kb_math

    # 预览模式（只验证数据，不存储）
    python scripts/import_chunks.py --chunks chunks.json --dry-run

    # 清理数据
    python scripts/import_chunks.py --clean-only --kb-id kb_math

    # 查询测试
    python scripts/import_chunks.py --query-only --kb-id kb_math --query "什么是珐琅"

    # 查看样例数据
    python scripts/import_chunks.py --sample-only --kb-id kb_math

    # 使用自定义 embedding URL
    python scripts/import_chunks.py --chunks chunks.json --kb-id kb_test --embedding-url http://192.168.8.233:11434
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent))

# from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db
from app.core.database import get_database, mongodb
from app.core.config import settings
# from app.services.rag_service import RAGRetrieval
from app.utils.embeddings import get_embedding
from app.core.logging import get_logger

logger = get_logger(__name__)


# Test queries covering different scenarios
TEST_QUERIES = [
    {
        "type": "简单概念",
        "query": "什么是珐琅工艺",
        "description": "基础知识召回",
    },
    {
        "type": "具体工艺",
        "query": "掐丝珐琅的制作步骤",
        "description": "流程描述召回",
    },
    {
        "type": "材料相关",
        "query": "适合烧制珐琅的金属",
        "description": "细节信息召回",
    },
    {
        "type": "历史背景",
        "query": "珐琅工艺的起源",
        "description": "背景知识召回",
    },
    {
        "type": "比较类",
        "query": "掐丝珐琅和画珐琅的区别",
        "description": "跨章节信息召回",
    },
    {
        "type": "工具设备",
        "query": "珐琅制作需要哪些工具",
        "description": "列表类信息召回",
    },
]


@dataclass
class ImportChunk:
    """导入的 chunk 数据结构"""
    chunk_id: str
    doc_id: str
    kb_id: str
    content: str
    chunk_index: int
    title_path: List[str]
    page_idx: int
    page_indices: List[int]
    has_definition: bool
    has_example: bool
    has_formula: bool
    has_image: bool
    has_table: bool
    has_list: bool
    image_references: List[str]
    image_captions: List[str]
    structure_level: int
    raw_data: Dict[str, Any]


class ChunkImporter:
    """通用 chunk 数据导入器"""

    def __init__(
        self,
        chunks_file: str,
        kb_id: str,
        embedding_url: Optional[str] = None,
    ):
        self.chunks_file = chunks_file
        self.kb_id = kb_id
        self.embedding_url = embedding_url

        # Generate a unique doc_id for this import
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.doc_id = f"doc_import_{timestamp}"

        # Storage for loaded chunks
        self.chunks: List[ImportChunk] = []

        # Results storage
        self.results: Dict[str, Any] = {
            "chunks_file": chunks_file,
            "kb_id": self.kb_id,
            "doc_id": self.doc_id,
            "timestamp": datetime.now().isoformat(),
        }

    async def setup(self):
        """Setup database connections."""
        print("\n" + "=" * 80)
        print("🚀 Chunk Import Tool")
        print("=" * 80)

        print(f"\n📋 Import Configuration:")
        print(f"  Chunks File: {self.chunks_file}")
        print(f"  KB ID: {self.kb_id}")
        print(f"  Doc ID: {self.doc_id}")

        # Get actual embedding URL (from args or config)
        actual_embedding_url = self.embedding_url
        print(f"  Embedding URL: {actual_embedding_url}")

        # Initialize database connection
        print(f"\n🔌 Connecting to databases...")
        try:
            await mongodb.connect()
            print(f"  ✅ MongoDB connected")
        except Exception as e:
            print(f"  ⚠️ MongoDB connection failed: {e}")

        try:
            chroma_db.connect()
            print(f"  ✅ ChromaDB connected")
        except Exception as e:
            print(f"  ⚠️ ChromaDB connection failed: {e}")

        try:
            await es_db.connect()
            print(f"  ✅ ElasticSearch connected")
        except Exception as e:
            print(f"  ⚠️ ElasticSearch connection failed: {e}")

        # Ensure knowledge base exists
        await self._ensure_kb_exists()

    async def _ensure_kb_exists(self):
        """Ensure knowledge base exists in MongoDB, create if not."""
        print(f"\n📚 Checking knowledge base...")

        try:
            db = await get_database()
            kb = await db.knowledge_bases.find_one({"kb_id": self.kb_id})

            if not kb:
                # Create knowledge base record, use kb_id as name
                from app.models.database import KnowledgeBaseModel

                kb_model = KnowledgeBaseModel(
                    kb_id=self.kb_id,
                    name=self.kb_id,
                    description=f"{self.kb_id}知识库 - 用于 chunk 导入",
                    status="active",
                )
                await db.knowledge_bases.insert_one(kb_model.model_dump())
                print(f"  ✅ Created knowledge base: {self.kb_id}")
            else:
                existing_name = kb.get("name", "")
                print(f"  ✅ Using existing knowledge base: {existing_name} ({self.kb_id})")

        except Exception as e:
            print(f"  ⚠️ Knowledge base check failed: {e}")

    async def load_chunks(self) -> List[Dict]:
        """加载 chunks.json 格式的数据"""
        print("\n" + "-" * 80)
        print("📄 Loading Chunks")
        print("-" * 80)

        try:
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                raw_chunks = json.load(f)

            if not isinstance(raw_chunks, list):
                raise ValueError("Expected a JSON array of chunks")

            print(f"  ✅ Loaded {len(raw_chunks)} chunks from file")

            self.results["raw_chunk_count"] = len(raw_chunks)
            return raw_chunks

        except FileNotFoundError:
            print(f"  ❌ File not found: {self.chunks_file}")
            raise
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parse error: {e}")
            raise

    def _transform_chunk(self, raw_chunk: Dict, index: int) -> ImportChunk:
        """
        Transform textbook-chunker chunk format to ImportChunk format.

        Input format:
        {
            "id": "knowledge_9",
            "text": "content",
            "metadata": {
                "title_path": ["title"],
                "main_title": "title",
                "level": 1,
                "page_start": 1,
                "page_end": 1,
                "has_definition": false,
                ...
            },
            "raw": {
                "images": [{"img_path": "...", "caption": "...", "page": 1}],
                ...
            }
        }
        """
        metadata = raw_chunk.get("metadata", {})
        raw = raw_chunk.get("raw", {})

        # Extract image references and captions
        image_references = []
        image_captions = []
        for img in raw.get("images", []):
            img_path = img.get("img_path", "")
            caption = img.get("caption", "")
            if img_path:
                image_references.append(img_path)
            if caption:
                image_captions.append(caption)

        # Generate chunk_id
        raw_id = raw_chunk.get("id")
        if raw_id:
            chunk_id = raw_id
        else:
            chunk_id = f"chunk_{self.doc_id}_{index}"

        # Page indices
        page_start = metadata.get("page_start", 0)
        page_end = metadata.get("page_end", page_start)
        if page_start > 0:
            page_indices = list(range(page_start, page_end + 1))
        else:
            page_indices = []

        return ImportChunk(
            chunk_id=chunk_id,
            doc_id=self.doc_id,
            kb_id=self.kb_id,
            content=raw_chunk.get("text", ""),
            chunk_index=index,
            title_path=metadata.get("title_path", []),
            page_idx=page_start,
            page_indices=page_indices,
            has_definition=metadata.get("has_definition", False),
            has_example=metadata.get("has_example", False),
            has_formula=metadata.get("has_formula", False),
            has_image=metadata.get("has_image", False),
            has_table=metadata.get("has_table", False),
            has_list=metadata.get("has_list", False),
            image_references=image_references,
            image_captions=image_captions,
            structure_level=metadata.get("level", 0),
            raw_data=raw_chunk,
        )

    async def step1_validate_chunks(self) -> bool:
        """Step 1: 验证 chunk 数据"""
        print("\n" + "-" * 80)
        print("📋 Step 1: Validating Chunks")
        print("-" * 80)

        try:
            raw_chunks = await self.load_chunks()

            # Transform chunks
            self.chunks = []
            for i, raw_chunk in enumerate(raw_chunks):
                chunk = self._transform_chunk(raw_chunk, i)
                self.chunks.append(chunk)

            print(f"  ✅ Transformed {len(self.chunks)} chunks")

            # Statistics
            total_chars = sum(len(c.content) for c in self.chunks)
            avg_chars = total_chars // len(self.chunks) if self.chunks else 0
            chunks_with_titles = sum(1 for c in self.chunks if c.title_path)
            chunks_with_images = sum(1 for c in self.chunks if c.image_references)
            chunks_with_definitions = sum(1 for c in self.chunks if c.has_definition)
            chunks_with_examples = sum(1 for c in self.chunks if c.has_example)
            chunks_with_formulas = sum(1 for c in self.chunks if c.has_formula)
            chunks_with_tables = sum(1 for c in self.chunks if c.has_table)
            chunks_with_lists = sum(1 for c in self.chunks if c.has_list)

            print(f"  ✅ Total characters: {total_chars}")
            print(f"  ✅ Average chunk size: {avg_chars} chars")
            print(f"  ✅ Chunks with titles: {chunks_with_titles}")
            print(f"  ✅ Chunks with images: {chunks_with_images}")
            print(f"  ✅ Chunks with definitions: {chunks_with_definitions}")
            print(f"  ✅ Chunks with examples: {chunks_with_examples}")
            print(f"  ✅ Chunks with formulas: {chunks_with_formulas}")
            print(f"  ✅ Chunks with tables: {chunks_with_tables}")
            print(f"  ✅ Chunks with lists: {chunks_with_lists}")

            # Show sample chunks
            print(f"\n  📝 Sample chunks (first 3):")
            for i, chunk in enumerate(self.chunks[:3]):
                content_preview = chunk.content[:100].replace('\n', ' ')
                print(f"    [{i+1}] {chunk.chunk_id}")
                print(f"        Title: {' > '.join(chunk.title_path) if chunk.title_path else 'N/A'}")
                print(f"        Content: {content_preview}...")
                print()

            self.results["validation"] = {
                "total_chunks": len(self.chunks),
                "total_chars": total_chars,
                "avg_chars": avg_chars,
                "chunks_with_titles": chunks_with_titles,
                "chunks_with_images": chunks_with_images,
                "chunks_with_definitions": chunks_with_definitions,
                "chunks_with_examples": chunks_with_examples,
                "chunks_with_formulas": chunks_with_formulas,
                "chunks_with_tables": chunks_with_tables,
                "chunks_with_lists": chunks_with_lists,
            }

            return True

        except Exception as e:
            print(f"  ❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def step2_clear_test_data(self) -> bool:
        """Step 2: 清理旧数据 (by kb_id)"""
        print("\n" + "-" * 80)
        print("🧹 Step 2: Clearing Existing Data (by kb_id)")
        print("-" * 80)

        try:
            db = await get_database()

            # Delete from MongoDB by kb_id
            doc_result = await db.documents.delete_many({"kb_id": self.kb_id})
            chunk_result = await db.document_chunks.delete_many({"kb_id": self.kb_id})
            print(f"  ✅ MongoDB: deleted {doc_result.deleted_count} docs, {chunk_result.deleted_count} chunks")

            # Delete from ChromaDB by kb_id
            if chroma_db.client:
                try:
                    chroma_db.doc_collection.delete(where={"kb_id": self.kb_id})
                    print(f"  ✅ ChromaDB: deleted data for kb_id={self.kb_id}")
                except Exception as e:
                    print(f"  ⚠️ ChromaDB delete: {e}")

            # Delete from ElasticSearch by kb_id
            try:
                await es_db.delete_by_query(
                    index='doc',
                    body={"query": {"term": {"kb_id": self.kb_id}}},
                )
                print(f"  ✅ ElasticSearch: deleted data for kb_id={self.kb_id}")
            except Exception as e:
                print(f"  ⚠️ ElasticSearch delete: {e}")

            return True

        except Exception as e:
            print(f"  ❌ Clear failed: {e}")
            return False

    async def step3_store_and_vectorize(self) -> bool:
        """Step 3: 存储和向量化"""
        print("\n" + "-" * 80)
        print("💾 Step 3: Storing and Vectorizing")
        print("-" * 80)

        try:
            # Get embedder
            embedder = get_embedding()

            print(f"  📝 Using embedder: {type(embedder).__name__}")
            print(f"  📝 Model: {getattr(embedder, 'model', 'N/A')}")

            # Prepare data for ChromaDB
            chunk_texts = []
            chunk_ids = []
            chunk_metadatas = []

            for chunk in self.chunks:
                # Use title-prefixed content for embedding
                if chunk.title_path:
                    title_str = " > ".join(chunk.title_path)
                    chunk_texts.append(f"{title_str}\n\n{chunk.content}")
                else:
                    chunk_texts.append(chunk.content)

                chunk_ids.append(chunk.chunk_id)
                chunk_metadatas.append({
                    "doc_id": chunk.doc_id,
                    "kb_id": chunk.kb_id,
                    "chunk_index": chunk.chunk_index,
                    "page_idx": chunk.page_idx,
                    "has_images": len(chunk.image_references) > 0,
                    "has_definition": chunk.has_definition,
                    "has_example": chunk.has_example,
                    "has_formula": chunk.has_formula,
                    "structure_level": chunk.structure_level,
                })

            print(f"  📝 Vectorizing {len(chunk_texts)} chunks...")

            # Test single embedding first
            print(f"  📝 Testing single embedding...")
            try:
                test_embedding = embedder.embed_query("测试文本")
                print(f"  ✅ Test embedding dimension: {len(test_embedding)}")
            except Exception as e:
                print(f"  ❌ Test embedding failed: {e}")
                return False

            # Store in ChromaDB
            if chroma_db.client:
                await chroma_db.add_documents(
                    collection_name="doc",
                    documents=chunk_texts,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids,
                )
                print(f"  ✅ ChromaDB: stored {len(chunk_ids)} chunks")

            # Store in ElasticSearch
            for chunk in self.chunks:
                es_document = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "kb_id": chunk.kb_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "page_idx": chunk.page_idx,
                    "image_count": len(chunk.image_references),
                    "has_definition": chunk.has_definition,
                    "has_example": chunk.has_example,
                    "has_formula": chunk.has_formula,
                    "has_table": chunk.has_table,
                    "has_list": chunk.has_list,
                }

                await es_db.index_document(
                    index="doc",
                    doc_id=chunk.chunk_id,
                    document=es_document,
                )

            print(f"  ✅ ElasticSearch: stored {len(self.chunks)} chunks")

            # Store in MongoDB
            db = await get_database()
            chunk_docs = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "kb_id": c.kb_id,
                    "content": c.content,
                    "chunk_index": c.chunk_index,
                    "page_idx": c.page_idx,
                    "page_indices": c.page_indices,
                    "image_references": c.image_references,
                    "image_captions": c.image_captions,
                    "title_path": c.title_path,
                    "structure_level": c.structure_level,
                    "has_definition": c.has_definition,
                    "has_example": c.has_example,
                    "has_formula": c.has_formula,
                    "has_image": c.has_image,
                    "has_table": c.has_table,
                    "has_list": c.has_list,
                    "metadata": {
                        "main_title": c.raw_data.get("metadata", {}).get("main_title"),
                        "level": c.structure_level,
                    },
                }
                for c in self.chunks
            ]

            await db.document_chunks.insert_many(chunk_docs)
            print(f"  ✅ MongoDB: stored {len(chunk_docs)} chunks")

            # Also create a document record
            from app.models.database import DocumentModel
            doc_model = DocumentModel(
                doc_id=self.doc_id,
                kb_id=self.kb_id,
                filename=Path(self.chunks_file).name,
                chunks_count=len(self.chunks),
                vectors_count=len(self.chunks),
                status="completed",
                format="JSON",
                size=Path(self.chunks_file).stat().st_size if Path(self.chunks_file).exists() else 0,
                metadata={
                    "imported_from": "chunks.json",
                    "imported_at": datetime.now().isoformat(),
                },
            )
            await db.documents.insert_one(doc_model.model_dump())
            print(f"  ✅ MongoDB: stored document record")

            return True

        except Exception as e:
            print(f"  ❌ Storage failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def step4_retrieve_and_score(self) -> bool:
        """Step 4: 检索和评分"""
        print("\n" + "-" * 80)
        print("🔍 Step 4: Retrieval and Scoring")
        print("-" * 80)

        try:
            retrieval_results = []

            for test_case in TEST_QUERIES:
                print(f"\n  📝 Query Type: {test_case['type']}")
                print(f"     Query: {test_case['query']}")

                # Retrieve using RAG service
                rag = RAGRetrieval()
                results = await rag.search(
                    query=test_case['query'],
                    kb_ids=[self.kb_id],
                    top_k=10,
                )

                print(f"     Retrieved: {len(results)} results")

                # Show top results
                for i, doc in enumerate(results[:3]):
                    content_preview = doc.get('content', '')[:100]
                    score = doc.get('rrf_score', doc.get('score', 0))
                    has_images = doc.get('image_count', 0) > 0
                    title_path = doc.get('title_path', [])

                    print(f"       [{i+1}] Score: {score:.4f} | Images: {has_images}")
                    if title_path:
                        print(f"           Path: {' > '.join(title_path)}")
                    print(f"           Content: {content_preview}...")

                retrieval_results.append({
                    "type": test_case['type'],
                    "query": test_case['query'],
                    "results_count": len(results),
                    "top_scores": [r.get('rrf_score', r.get('score', 0)) for r in results[:3]],
                })

            self.results["retrieval"] = retrieval_results

            return True

        except Exception as e:
            print(f"  ❌ Retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def run(self, dry_run: bool = False) -> bool:
        """运行完整导入流程"""
        await self.setup()

        # Step 1: Validate
        if not await self.step1_validate_chunks():
            await self._cleanup_databases()
            return False

        if dry_run:
            print("\n⚠️ Dry run mode: skipping storage and retrieval")
            await self._cleanup_databases()
            return True

        # Step 2: Clear test data
        if not await self.step2_clear_test_data():
            await self._cleanup_databases()
            return False

        # Step 3: Store and vectorize
        if not await self.step3_store_and_vectorize():
            await self._cleanup_databases()
            return False

        # Step 4: Retrieve and score
        if not await self.step4_retrieve_and_score():
            await self._cleanup_databases()
            return False

        # Print summary
        print("\n" + "=" * 80)
        print("📊 Import Summary")
        print("=" * 80)
        print(f"  Doc ID: {self.doc_id}")
        print(f"  KB ID: {self.kb_id}")
        print(f"  Chunks imported: {len(self.chunks)}")

        # Disconnect databases
        await self._cleanup_databases()

        return True

    async def _cleanup_databases(self):
        """Disconnect from databases."""
        try:
            await mongodb.disconnect()
        except Exception:
            pass


async def clean_only_mode(kb_id: str):
    """只清理数据模式"""
    print("\n" + "=" * 80)
    print("🧹 Clean Only Mode")
    print("=" * 80)
    print(f"KB ID: {kb_id}")

    # 连接数据库
    try:
        await mongodb.connect()
        print(f"  ✅ MongoDB connected")
    except Exception as e:
        print(f"  ⚠️ MongoDB connection failed: {e}")

    try:
        chroma_db.connect()
        print(f"  ✅ ChromaDB connected")
    except Exception as e:
        print(f"  ⚠️ ChromaDB connection failed: {e}")

    try:
        await es_db.connect()
        print(f"  ✅ ElasticSearch connected")
    except Exception as e:
        print(f"  ⚠️ ElasticSearch connection failed: {e}")

    # 执行清理
    print("\n" + "-" * 80)
    print("🧹 Cleaning Data (by kb_id)")
    print("-" * 80)

    try:
        db = await get_database()

        # MongoDB
        doc_result = await db.documents.delete_many({"kb_id": kb_id})
        chunk_result = await db.document_chunks.delete_many({"kb_id": kb_id})
        print(f"  ✅ MongoDB: deleted {doc_result.deleted_count} docs, {chunk_result.deleted_count} chunks")

        # ChromaDB
        if chroma_db.client:
            try:
                chroma_db.doc_collection.delete(where={"kb_id": kb_id})
                print(f"  ✅ ChromaDB: deleted data for kb_id={kb_id}")
            except Exception as e:
                print(f"  ⚠️ ChromaDB delete: {e}")

        # ElasticSearch
        try:
            await es_db.delete_by_query(
                index='doc',
                body={"query": {"term": {"kb_id": kb_id}}},
            )
            print(f"  ✅ ElasticSearch: deleted data for kb_id={kb_id}")
        except Exception as e:
            print(f"  ⚠️ ElasticSearch delete: {e}")

        print("\n✅ Clean completed!")
        await mongodb.disconnect()
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Clean failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def query_only_mode(kb_id: str, query: Optional[str] = None, top_k: int = 10, rerank_enabled: bool = True):
    """只检索查询模式"""
    print("\n" + "=" * 80)
    print("🔍 Query Only Mode")
    print("=" * 80)
    print(f"KB ID: {kb_id}")

    rerank_status = "启用" if rerank_enabled else "禁用"
    print(f"Rerank: {rerank_status}")

    # 连接数据库
    try:
        await mongodb.connect()
        chroma_db.connect()
        await es_db.connect()
        print(f"  ✅ All databases connected")
    except Exception as e:
        print(f"  ⚠️ Database connection warning: {e}")

    # 确定查询列表
    if query:
        queries = [{"type": "自定义", "query": query, "description": "用户自定义查询"}]
    else:
        queries = TEST_QUERIES

    print(f"\n📝 Running {len(queries)} queries...\n")

    try:
        for test_case in queries:
            print("-" * 80)
            print(f"📝 Query Type: {test_case['type']}")
            print(f"   Query: {test_case['query']}")

            rag = RAGRetrieval()
            results = await rag.search(
                query=test_case['query'],
                kb_ids=[kb_id],
                top_k=top_k,
                enable_rerank=rerank_enabled,
            )

            print(f"   Retrieved: {len(results)} results\n")

            # 显示所有结果
            for i, doc in enumerate(results):
                content_preview = doc.get('content', '')[:150]
                score = doc.get('rrf_score', doc.get('score', 0))
                has_images = doc.get('image_count', 0) > 0
                title_path = doc.get('title_path', [])

                print(f"   [{i+1}] Score: {score:.4f} | Images: {has_images}")
                if title_path:
                    print(f"       Path: {' > '.join(title_path)}")
                print(f"       Content: {content_preview}...")
                print()

        await mongodb.disconnect()
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def sample_only_mode(kb_id: str):
    """查看样例数据模式"""
    print("\n" + "=" * 80)
    print("📊 Sample Data Mode")
    print("=" * 80)
    print(f"KB ID: {kb_id}")

    # 连接数据库
    try:
        await mongodb.connect()
        chroma_db.connect()
        await es_db.connect()
        print(f"  ✅ All databases connected")
    except Exception as e:
        print(f"  ⚠️ Database connection warning: {e}")

    print("\n" + "-" * 80)
    print("📦 MongoDB Sample Data")
    print("-" * 80)

    try:
        db = await get_database()

        # 统计
        doc_count = await db.documents.count_documents({"kb_id": kb_id})
        chunk_count = await db.document_chunks.count_documents({"kb_id": kb_id})
        print(f"  📄 Documents: {doc_count}")
        print(f"  📋 Chunks: {chunk_count}")

        # 样例 chunk
        if chunk_count > 0:
            sample_chunks = await db.document_chunks.find({"kb_id": kb_id}).limit(3).to_list(None)
            print(f"\n  Sample chunks (first 3):")
            for i, chunk in enumerate(sample_chunks):
                content_preview = chunk.get('content', '')[:200]
                print(f"    [{i+1}] chunk_id: {chunk.get('chunk_id')}")
                print(f"        doc_id: {chunk.get('doc_id')}")
                print(f"        page_idx: {chunk.get('page_idx')}")
                print(f"        content: {content_preview}...")
                print()
    except Exception as e:
        print(f"  ⚠️ MongoDB query failed: {e}")

    print("\n" + "-" * 80)
    print("📦 ChromaDB Sample Data")
    print("-" * 80)

    try:
        if chroma_db.client and chroma_db.doc_collection:
            count = chroma_db.doc_collection.count()
            print(f"  📋 Total chunks in collection: {count}")

            # 获取样例
            results = chroma_db.doc_collection.get(
                where={"kb_id": kb_id},
                limit=3
            )
            print(f"  📋 Chunks for kb_id={kb_id}: {len(results.get('ids', []))}")

            if results.get('ids'):
                print(f"\n  Sample chunks (first 3):")
                for i, chunk_id in enumerate(results['ids'][:3]):
                    doc = results['documents'][i] if i < len(results['documents']) else ""
                    metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
                    print(f"    [{i+1}] id: {chunk_id}")
                    print(f"        metadata: {metadata}")
                    print(f"        content: {doc[:200]}...")
                    print()
    except Exception as e:
        print(f"  ⚠️ ChromaDB query failed: {e}")

    print("\n" + "-" * 80)
    print("📦 ElasticSearch Sample Data")
    print("-" * 80)

    try:
        # 统计
        count_result = await es_db.client.count(index='doc', body={
            "query": {"term": {"kb_id": kb_id}}
        })
        print(f"  📋 Documents for kb_id={kb_id}: {count_result['count']}")

        # 获取样例
        search_result = await es_db.client.search(index='doc', body={
            "query": {"term": {"kb_id": kb_id}},
            "size": 3
        })

        if search_result['hits']['hits']:
            print(f"\n  Sample documents (first 3):")
            for i, hit in enumerate(search_result['hits']['hits']):
                source = hit['_source']
                print(f"    [{i+1}] id: {hit['_id']}")
                print(f"        chunk_id: {source.get('chunk_id')}")
                print(f"        page_idx: {source.get('page_idx')}")
                content = source.get('content', '')[:200]
                print(f"        content: {content}...")
                print()
    except Exception as e:
        print(f"  ⚠️ ElasticSearch query failed: {e}")

    await mongodb.disconnect()
    sys.exit(0)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="通用 Chunk 数据导入脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 完整导入流程
  %(prog)s --chunks chunks.json --kb-id kb_math

  # 预览模式（只验证数据）
  %(prog)s --chunks chunks.json --dry-run

  # 清理数据
  %(prog)s --clean-only --kb-id kb_math

  # 只检索查询（使用预定义查询）
  %(prog)s --query-only --kb-id kb_math

  # 自定义查询
  %(prog)s --query-only --kb-id kb_math --query "什么是珐琅"

  # 查看样例数据
  %(prog)s --sample-only --kb-id kb_math

  # 使用自定义 embedding URL
  %(prog)s --chunks chunks.json --kb-id kb_test --embedding-url http://192.168.8.233:11434
        """
    )

    parser.add_argument(
        "--chunks",
        help="Chunks JSON file path (导入模式必需)"
    )
    parser.add_argument(
        "--embedding-url",
        help="Custom embedding URL (overrides config)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate chunks, skip storage"
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean data by kb_id, then exit"
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Only run retrieval queries, skip import"
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Show sample data from databases by kb_id"
    )
    parser.add_argument(
        "--query",
        help="Custom query string for --query-only mode"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable reranking for retrieval queries (default: enabled)"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking for retrieval queries"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--kb-id",
        default="kb_5f2a02bd5dfe",
        help="Knowledge base ID (default: kb_5f2a02bd5dfe)"
    )

    args = parser.parse_args()

    # --clean-only 模式
    if args.clean_only:
        await clean_only_mode(args.kb_id)

    # --query-only 模式
    if args.query_only:
        # 确定 rerank 设置（默认启用）
        rerank_enabled = True
        if args.no_rerank:
            rerank_enabled = False
        elif args.rerank:
            rerank_enabled = True

        await query_only_mode(args.kb_id, args.query, args.top_k, rerank_enabled)

    # --sample-only 模式
    if args.sample_only:
        await sample_only_mode(args.kb_id)

    # 完整导入模式（需要 chunks 文件）
    if not args.chunks:
        print(f"❌ Error: --chunks is required for import mode", file=sys.stderr)
        sys.exit(1)

    # Check input file
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"❌ Error: File not found: {args.chunks}", file=sys.stderr)
        sys.exit(1)

    # Run import
    importer = ChunkImporter(
        chunks_file=str(chunks_path),
        kb_id=args.kb_id,
        embedding_url=args.embedding_url,
    )

    try:
        success = await importer.run(dry_run=args.dry_run)

        # Save results if requested
        if args.output and hasattr(importer, 'results'):
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(importer.results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Results saved to: {args.output}")

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Import interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
