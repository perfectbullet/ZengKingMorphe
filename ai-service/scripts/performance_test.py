"""
Performance testing script for AI service.
Tests query latency with caching enabled.
"""
import time
import requests
import json
from typing import Dict, List


class PerformanceTester:
    """Test AI service performance"""

    def __init__(self, base_url: str = "http://localhost:8100"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat/v1/chat/completions"

    def send_query(self, query: str, employee_id: str = "hutao") -> Dict:
        """
        Send a query and measure timing.

        Note: The API only supports streaming responses.
        The stream is consumed internally and timing is measured.
        """

        payload = {
            "messages": [{"role": "user", "content": query}],
            "employee_id": employee_id,
            "user_id": "perf_test_user",
            "session_id": "perf_test_session",
            "stream": True  # API only supports streaming
        }

        start_time = time.time()

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=30.0
            )
            response.raise_for_status()

            # Consume the stream
            full_content = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line or line.startswith(":"):
                    continue
                if line == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    chunk_data = json.loads(line)
                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                        delta = chunk_data["choices"][0].get("delta", {})
                        if "content" in delta:
                            full_content += delta["content"]
                except json.JSONDecodeError:
                    pass

            duration_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "duration_ms": duration_ms,
                "status_code": response.status_code,
                "query": query,
                "response_length": len(full_content)
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "duration_ms": duration_ms,
                "error": str(e),
                "query": query
            }

    def test_scenario(self, queries: List[str], wait_seconds: int = 180) -> List[Dict]:
        """
        Test a scenario with multiple queries.

        Args:
            queries: List of queries to test
            wait_seconds: Seconds to wait between tests (for cache expiry)

        Returns:
            List of timing results
        """
        results = []

        print(f"\n{'='*60}")
        print(f"Testing {len(queries)} queries with {wait_seconds}s wait interval")
        print(f"{'='*60}\n")

        for i, query in enumerate(queries, 1):
            print(f"Test {i}: '{query[:50]}...'")

            # First query
            print(f"  Sending first query...")
            result1 = self.send_query(query)
            results.append(result1)

            if result1["success"]:
                print(f"  ✓ First query: {result1['duration_ms']:.0f}ms")
            else:
                print(f"  ✗ First query failed: {result1.get('error')}")
                continue

            # Wait for keep-alive interval
            if wait_seconds > 0:
                print(f"  Waiting {wait_seconds}s for cache/keep-alive...")
                time.sleep(wait_seconds)

            # Second query
            print(f"  Sending second query...")
            result2 = self.send_query(query)
            results.append(result2)

            if result2["success"]:
                improvement = (1 - result2['duration_ms'] / result1['duration_ms']) * 100
                print(f"  ✓ Second query: {result2['duration_ms']:.0f}ms (improvement: {improvement:.1f}%)")
            else:
                print(f"  ✗ Second query failed: {result2.get('error')}")

            print()

        return results

    def print_summary(self, results: List[Dict]):
        """Print performance summary"""

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if not successful:
            print("\n❌ All queries failed!")
            return

        durations = [r["duration_ms"] for r in successful]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Total queries: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print()
        print(f"Average duration: {avg_duration:.0f}ms")
        print(f"Min duration: {min_duration:.0f}ms")
        print(f"Max duration: {max_duration:.0f}ms")
        print()

        # Calculate improvement for pairs
        if len(successful) >= 2:
            for i in range(0, len(successful) - 1, 2):
                r1 = successful[i]
                r2 = successful[i + 1]
                improvement = (1 - r2['duration_ms'] / r1['duration_ms']) * 100
                print(f"Query pair {i//2 + 1}: {improvement:+.1f}% improvement")

        print(f"{'='*60}\n")

    def test_embedding_cache(self, query: str, repeat: int = 5):
        """
        Test embedding cache effectiveness by repeating the same query.

        Args:
            query: Query to repeat
            repeat: Number of repetitions
        """
        print(f"\n{'='*60}")
        print(f"Testing embedding cache with {repeat} repetitions")
        print(f"Query: '{query}'")
        print(f"{'='*60}\n")

        results = []
        for i in range(repeat):
            result = self.send_query(query)
            results.append(result)

            if result["success"]:
                print(f"  Repetition {i+1}: {result['duration_ms']:.0f}ms")
            else:
                print(f"  Repetition {i+1}: FAILED")

            # Small delay between requests
            time.sleep(0.5)

        # Calculate cache improvement
        successful = [r for r in results if r["success"]]
        if len(successful) >= 2:
            first = successful[0]["duration_ms"]
            avg_subsequent = sum(r["duration_ms"] for r in successful[1:]) / (len(successful) - 1)
            improvement = (1 - avg_subsequent / first) * 100

            print()
            print(f"First query: {first:.0f}ms")
            print(f"Average subsequent: {avg_subsequent:.0f}ms")
            print(f"Cache improvement: {improvement:.1f}%")

        print()


def main():
    """Main test runner"""

    tester = PerformanceTester()

    # Test 1: Repeated query (embedding cache test)
    print("\n" + "🧪 "*20)
    print("TEST 1: Embedding Cache Effectiveness")
    print("🧪 "*20)
    tester.test_embedding_cache("失蜡铸造的原理", repeat=5)

    # Test 2: Different queries (keep-alive test)
    print("\n" + "🧪 "*20)
    print("TEST 2: Ollama Keep-Alive (3-minute interval)")
    print("🧪 "*20)
    queries = [
        "游标卡尺的用法",
        "金属浇铸的步骤"
    ]
    tester.test_scenario(queries, wait_seconds=180)

    # Test 3: Quick succession (model stays loaded)
    print("\n" + "🧪 "*20)
    print("TEST 3: Model Stay-Loaded (no wait)")
    print("🧪 "*20)
    tester.test_scenario(queries, wait_seconds=0)

    print("\n" + "✅ "*20)
    print("All tests completed!")
    print("✅ "*20 + "\n")


if __name__ == "__main__":
    main()
