from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable

def run_parallel(jobs: List[Dict,], fn: Callable[[Dict[str, Any]], Dict[str, Any]], max_workers: int = 4):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(fn, job) for job in jobs]
        for fut in as_completed(futs):
            results.append(fut.result())
    return results
