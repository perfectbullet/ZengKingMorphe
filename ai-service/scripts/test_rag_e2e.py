#!/usr/bin/env python3
"""
RAG End-to-End Test Script

Tests the complete RAG pipeline:
1. Parse MinerU JSON
2. Chunk with structure-aware strategy
3. Store in vector databases (ChromaDB + ElasticSearch)
4. Retrieve with test queries
5. Score and evaluate results

Usage:
    # Basic test (uses default knowledge base)
    python scripts/test_rag_e2e.py --json input.json

    # With custom knowledge base
    python scripts/test_rag_e2e.py --json input.json --kb-id kb_5f2a02bd5dfe --kb-name "首饰设计"

    # With custom chunk size
    python scripts/test_rag_e2e.py --json input.json --chunk-size 1000

    # Skip storage (just parsing + chunking)
    python scripts/test_rag_e2e.py --json input.json --dry-run

    # Clean test data after run
    python scripts/test_rag_e2e.py --json input.json --clean-after

    # Use specific embedding URL
    python scripts/test_rag_e2e.py --json input.json --embedding-url http://192.168.8.233:11434
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.mineru_json_parser import MinerUJsonParser
from app.services.mineru_aware_chunking import MinerUAwareChunker, ChunkingStrategy, MultiModalChunk
from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db
from app.core.database import get_database
from app.core.config import settings
from app.services.rag_service import RAGRetrieval
from app.utils.embeddings import OllamaEmbeddings, get_embedding
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


class RAGE2ETester:
    """End-to-end RAG tester."""

    def __init__(
        self,
        json_file: str,
        kb_id: str = "kb_5f2a02bd5dfe",  # 使用用户指定的知识库ID
        kb_name: str = "首饰设计",           # 知识库名称
        chunk_size: int = 1000,
        strategy: str = "hybrid",
        embedding_url: Optional[str] = None,
        clean_after: bool = False,
    ):
        self.json_file = json_file
        self.kb_id = kb_id
        self.kb_name = kb_name
        self.chunk_size = chunk_size
        self.strategy = strategy
        self.embedding_url = embedding_url
        self.clean_after = clean_after

        # Test identifiers
        self.test_doc_id = f"test_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Results storage
        self.results: Dict[str, Any] = {
            "json_file": json_file,
            "kb_id": self.kb_id,
            "kb_name": self.kb_name,
            "test_doc_id": self.test_doc_id,
            "timestamp": datetime.now().isoformat(),
        }

    async def setup(self):
        """Setup test environment."""
        print("\n" + "=" * 80)
        print("🚀 RAG End-to-End Test")
        print("=" * 80)

        print(f"\n📋 Test Configuration:")
        print(f"  JSON File: {self.json_file}")
        print(f"  KB ID: {self.kb_id}")
        print(f"  KB Name: {self.kb_name}")
        print(f"  Doc ID: {self.test_doc_id}")
        print(f"  Chunk Size: {self.chunk_size}")
        print(f"  Strategy: {self.strategy}")
        if self.embedding_url:
            print(f"  Embedding URL: {self.embedding_url}")

        # Ensure knowledge base exists
        await self.ensure_kb_exists()

    async def ensure_kb_exists(self):
        """Ensure knowledge base exists in MongoDB, create if not."""
        print(f"\n📚 Checking knowledge base...")

        try:
            db = await get_database()
            kb = await db.knowledge_bases.find_one({"kb_id": self.kb_id})

            if not kb:
                # Create knowledge base record
                from app.models.database import KnowledgeBaseModel
                from datetime import datetime

                kb_model = KnowledgeBaseModel(
                    kb_id=self.kb_id,
                    name=self.kb_name,
                    description=f"{self.kb_name}知识库 - 用于RAG测试",
                    status="active",
                )
                await db.knowledge_bases.insert_one(kb_model.model_dump())
                print(f"  ✅ Created knowledge base: {self.kb_name} ({self.kb_id})")
            else:
                # Update existing knowledge base
                existing_name = kb.get("name", "")
                print(f"  ✅ Using existing knowledge base: {existing_name} ({self.kb_id})")

        except Exception as e:
            print(f"  ⚠️ Knowledge base check failed: {e}")
            # Continue anyway, as knowledge base may be in a different database

    async def step1_parse_json(self) -> bool:
        """Step 1: Parse MinerU JSON."""
        print("\n" + "-" * 80)
        print("📄 Step 1: Parsing MinerU JSON")
        print("-" * 80)

        try:
            parser = MinerUJsonParser()
            doc = parser.parse_file(self.json_file)

            print(f"  ✅ Parsed {doc.get_total_pages()} pages")

            # Count blocks
            total_blocks = sum(len(page.blocks) for page in doc.pdf_info)
            print(f"  ✅ Total blocks: {total_blocks}")

            # Count by type
            block_types = {}
            for page in doc.pdf_info:
                for block in page.blocks:
                    bt = block.block_type.value
                    block_types[bt] = block_types.get(bt, 0) + 1
            print(f"  ✅ Block types: {block_types}")

            self.results["parse_result"] = {
                "total_pages": doc.get_total_pages(),
                "total_blocks": total_blocks,
                "block_types": block_types,
            }

            return True

        except Exception as e:
            print(f"  ❌ Parse failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def step2_chunk_document(self) -> bool:
        """Step 2: Chunk document with structure-aware strategy."""
        print("\n" + "-" * 80)
        print("🔪 Step 2: Chunking Document")
        print("-" * 80)

        try:
            parser = MinerUJsonParser()
            doc = parser.parse_file(self.json_file)

            chunker = MinerUAwareChunker(
                max_chunk_size=self.chunk_size,
                strategy=self.strategy,
            )

            chunks = await chunker.chunk_document(
                doc=doc,
                doc_id=self.test_doc_id,
                kb_id=self.kb_id,
            )

            print(f"  ✅ Created {len(chunks)} chunks")

            # Statistics
            total_chars = sum(len(c.content) for c in chunks)
            avg_chars = total_chars // len(chunks) if chunks else 0
            chunks_with_titles = sum(1 for c in chunks if c.title_path)
            chunks_with_images = sum(1 for c in chunks if c.image_references)

            print(f"  ✅ Total characters: {total_chars}")
            print(f"  ✅ Average chunk size: {avg_chars} chars")
            print(f"  ✅ Chunks with titles: {chunks_with_titles}")
            print(f"  ✅ Chunks with images: {chunks_with_images}")

            self.results["chunks"] = [
                {
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "content_length": len(c.content),
                    "page_idx": c.page_idx,
                    "title_path": c.title_path,
                    "has_images": len(c.image_references) > 0,
                }
                for c in chunks
            ]

            self.chunks = chunks
            return True

        except Exception as e:
            print(f"  ❌ Chunking failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def step3_clear_test_data(self) -> bool:
        """Step 3: Clear existing test data."""
        print("\n" + "-" * 80)
        print("🧹 Step 3: Clearing Test Data")
        print("-" * 80)

        try:
            db = await get_database()

            # Delete from MongoDB
            doc_result = await db.documents.delete_many({"doc_id": self.test_doc_id})
            chunk_result = await db.document_chunks.delete_many({"doc_id": self.test_doc_id})
            print(f"  ✅ MongoDB: deleted {doc_result.deleted_count} docs, {chunk_result.deleted_count} chunks")

            # Delete from ChromaDB
            if chroma_db.client:
                try:
                    chroma_db.doc_collection.delete(where={"doc_id": self.test_doc_id})
                    print(f"  ✅ ChromaDB: deleted test data")
                except Exception as e:
                    print(f"  ⚠️ ChromaDB delete: {e}")

            # Delete from ElasticSearch
            try:
                await es_db.delete_by_query(
                    index="doc",
                    body={"query": {"term": {"doc_id": self.test_doc_id}}},
                )
                print(f"  ✅ ElasticSearch: deleted test data")
            except Exception as e:
                print(f"  ⚠️ ElasticSearch delete: {e}")

            return True

        except Exception as e:
            print(f"  ❌ Clear failed: {e}")
            return False

    async def step4_store_and_vectorize(self) -> bool:
        """Step 4: Store chunks and vectorize."""
        print("\n" + "-" * 80)
        print("💾 Step 4: Storing and Vectorizing")
        print("-" * 80)

        try:
            # Get embedder
            if self.embedding_url:
                embedder = OllamaEmbeddings(
                    model=settings.embedding_ollama_model,
                    base_url=self.embedding_url,
                    max_tokens=8192,
                )
            else:
                embedder = get_embedding()

            print(f"  📝 Using embedder: {type(embedder).__name__}")
            print(f"  📝 Model: {getattr(embedder, 'model', 'N/A')}")

            # Prepare data
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
                })

            print(f"  📝 Vectorizing {len(chunk_texts)} chunks...")

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
            for i, chunk in enumerate(self.chunks):
                es_document = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "kb_id": chunk.kb_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "page_idx": chunk.page_idx,
                    "image_count": len(chunk.image_references),
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
                    "image_references": c.image_references,
                    "title_path": c.title_path,
                }
                for c in self.chunks
            ]

            await db.document_chunks.insert_many(chunk_docs)
            print(f"  ✅ MongoDB: stored {len(chunk_docs)} chunks")

            return True

        except Exception as e:
            print(f"  ❌ Storage failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def step5_retrieve_and_score(self) -> bool:
        """Step 5: Retrieve with test queries and score."""
        print("\n" + "-" * 80)
        print("🔍 Step 5: Retrieval and Scoring")
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
                    top_k=5,
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

    async def cleanup(self):
        """Cleanup test data."""
        if self.clean_after:
            print("\n" + "-" * 80)
            print("🧹 Cleaning Up Test Data")
            print("-" * 80)

            await self.step3_clear_test_data()

    async def run(self, dry_run: bool = False) -> bool:
        """Run complete test."""
        await self.setup()

        # Step 1: Parse
        if not await self.step1_parse_json():
            return False

        # Step 2: Chunk
        if not await self.step2_chunk_document():
            return False

        if dry_run:
            print("\n⚠️ Dry run mode: skipping storage and retrieval")
            return True

        # Step 3: Clear test data
        if not await self.step3_clear_test_data():
            return False

        # Step 4: Store and vectorize
        if not await self.step4_store_and_vectorize():
            return False

        # Step 5: Retrieve and score
        if not await self.step5_retrieve_and_score():
            return False

        # Cleanup if requested
        await self.cleanup()

        # Print summary
        print("\n" + "=" * 80)
        print("📊 Test Summary")
        print("=" * 80)
        print(f"  Doc ID: {self.test_doc_id}")
        print(f"  KB ID: {self.kb_id}")
        print(f"  KB Name: {self.kb_name}")
        print(f"  Chunks created: {len(self.chunks)}")

        return True


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG End-to-End Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --json input.json
  %(prog)s --json input.json --kb-id kb_5f2a02bd5dfe --kb-name "首饰设计"
  %(prog)s --json input.json --chunk-size 1000 --strategy hybrid
  %(prog)s --json input.json --dry-run
  %(prog)s --json input.json --clean-after
  %(prog)s --json input.json --embedding-url http://192.168.8.233:11434
        """
    )

    parser.add_argument(
        "--json",
        required=True,
        help="MinerU JSON file path"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Max chunk size in characters (default: 1000)"
    )
    parser.add_argument(
        "--strategy",
        choices=["by_title", "by_page", "hybrid"],
        default="hybrid",
        help="Chunking strategy (default: hybrid)"
    )
    parser.add_argument(
        "--embedding-url",
        help="Custom embedding URL (overrides config)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse and chunk, skip storage"
    )
    parser.add_argument(
        "--clean-after",
        action="store_true",
        help="Clean test data after running"
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
    parser.add_argument(
        "--kb-name",
        default="首饰设计",
        help="Knowledge base name (default: 首饰设计)"
    )

    args = parser.parse_args()

    # Check input file
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"❌ Error: File not found: {args.json}", file=sys.stderr)
        sys.exit(1)

    # Run test
    tester = RAGE2ETester(
        json_file=str(json_path),
        kb_id=args.kb_id,
        kb_name=args.kb_name,
        chunk_size=args.chunk_size,
        strategy=args.strategy,
        embedding_url=args.embedding_url,
        clean_after=args.clean_after,
    )

    try:
        success = await tester.run(dry_run=args.dry_run)

        # Save results if requested
        if args.output and hasattr(tester, 'results'):
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(tester.results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Results saved to: {args.output}")

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
