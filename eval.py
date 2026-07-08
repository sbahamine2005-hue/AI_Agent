import json
import time
import uuid
from my_agent import graph, my_tools
from langchain_core.messages import HumanMessage


EVAL_MEMORY_THREAD = "eval_memory_thread"


def run_single_case(case: dict) -> dict:
    """
    Runs one test case through the LangGraph self-correction graph.
    Returns a result dict with pass/fail, tool used, attempts, and timing.
    """
    
    if case["category"] == "memory":
        thread_id = EVAL_MEMORY_THREAD
    else:
        thread_id = f"eval_thread_{case['id']}_{uuid.uuid4().hex[:6]}"

    config = {"configurable": {"thread_id": thread_id}}

    # Proper initial state — fixes the retry_count and best_score init issues
    initial_state = {
        "query": case["query"],
        "current_query": case["query"],
        "retry_count": 0,
        "best_score": -1.0,
    }

    start = time.time()
    try:
        result = graph.invoke(initial_state, config=config)
        elapsed = round(time.time() - start, 2)

        final_answer = result.get("best_answer") or result.get("messages", "")
        tool_used = result.get("tool_used", "unknown")
        attempts = result.get("retry_count", 0) + 1  # retry_count starts at 0
        score = result.get("best_score", 0.0)

        # Tool correctness — check if expected tool appears anywhere in the run
        tool_correct = case["expected_tool"] in tool_used or tool_used == case["expected_tool"]

        # Content correctness — all expected keywords present in answer
        answer_lower = final_answer.lower()
        content_correct = all(
            kw.lower() in answer_lower for kw in case.get("expected_contains", [])
        )

        passed = tool_correct and content_correct

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] #{case['id']} | Tool: {tool_used} | "
              f"Score: {score} | Attempts: {attempts} | Time: {elapsed}s")

        return {
            "id": case["id"],
            "category": case["category"],
            "query": case["query"][:60],
            "expected_tool": case["expected_tool"],
            "actual_tool": tool_used,
            "tool_correct": tool_correct,
            "content_correct": content_correct,
            "attempts": attempts,
            "best_score": score,
            "time_s": elapsed,
            "passed": passed,
        }

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"[ERROR] #{case['id']} — {str(e)}")
        return {
            "id": case["id"],
            "category": case["category"],
            "query": case["query"][:60],
            "passed": False,
            "error": str(e),
            "attempts": 0,
            "time_s": elapsed,
        }


def print_summary(results: list):
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    print(f"Overall pass rate: {passed}/{total} ({round(passed/total*100)}%)")

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, {"pass": 0, "total": 0})
        categories[cat]["total"] += 1
        if r.get("passed"):
            categories[cat]["pass"] += 1

    print("\nBy category:")
    for cat, counts in categories.items():
        rate = round(counts["pass"] / counts["total"] * 100)
        print(f"  {cat:<20} {counts['pass']}/{counts['total']} ({rate}%)")

    # Reformulation effectiveness — did retries actually help?
    multi_attempt = [r for r in results if r.get("attempts", 1) > 1]
    if multi_attempt:
        recovered = sum(1 for r in multi_attempt if r.get("passed"))
        print(f"\nReformulation recovery rate: "
              f"{recovered}/{len(multi_attempt)} cases that retried ended up passing")

    # Average score across all passed cases
    scored = [r["best_score"] for r in results if "best_score" in r]
    if scored:
        print(f"Average best_score: {round(sum(scored)/len(scored), 2)}/10")


def run_evaluation():
    with open("eval/test_cases.json", "r") as f:
        test_cases = json.load(f)

    results = []

    for i, case in enumerate(test_cases):
        if i > 0:                    # no sleep before first test
            time.sleep(30)    

        print(f"\n--- Test {case['id']}: {case['query'][:60]} ---")
        result = run_single_case(case)
        results.append(result)

        # Rate limiting — Groq free tier is strict
        # Memory tests chain together so no sleep between them
        if case["category"] != "memory":
            time.sleep(60)

    print_summary(results)

    with open("eval/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to eval/results.json")


if __name__ == "__main__":
    run_evaluation()