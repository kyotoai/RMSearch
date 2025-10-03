#!/usr/bin/env python3
"""Command-line wrapper derived from `train_en.ipynb`.

Each command mirrors a notebook section so experiment steps can be
re-run without opening the notebook. Parameters remain exposed as plain
module-level constants so that developers can tweak them without
modifying the procedural code blocks ported from the notebook.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Make sibling example modules importable (e.g. generate_tag_graph2.py)
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

# ---------------------------------------------------------------------------
# Global knobs copied from the notebook for easy experimentation.
# Edit these values to change behaviour; the command bodies intentionally
# match the original cells line-for-line aside from Python syntax fixes.
# ---------------------------------------------------------------------------

# Dataset / preprocessing parameters -------------------------------------------------
DATASET_NAME = "hatakeyama-llm-team/japanese2010"  # Example: "hatakeyama-llm-team/japanese2010"
DATASET_CONFIG: Optional[str] = None               # Example: "cosmopedia-v2"
DATA_SPLIT = "train"                                # Example: "train"
DATA_SAVE_NAME = "smollm-corpus"                   # Example output folder name
DATA_OUTPUT_ROOT = Path("./data")                  # Example: Path("./data")
N_SAMPLE_TRAIN = 100_000                            # Important: number of rows kept for train split
N_SAMPLE_TEST = 8_000                               # Important: size of held-out eval slice
N_SMALL_SAMPLE = 10_000                             # Important: size for quick experiments (df_small.csv)
RANDOM_SEED = 42                                    # Fix seed for reproducibility

# Tag graph / embedding parameters ---------------------------------------------------
WORKING_DIR = Path("/workspace/RMS_exp")           # Where intermediate files are stored
GEN_MODEL = os.environ.get("VLLM_MODEL", "/workspace/qwen7b")
EMB_MODEL = os.environ.get("EMB_MODEL", "intfloat/e5-mistral-7b-instruct")
N_GROUP = 5000                                      # Number of clusters for tag grouping
N_TAG_SAMPLE = 6                                    # Number of tags sampled when summarising a group
TENSOR_PARALLEL_SIZE = 1                            # Parallel config for embedding pool
NUM_INSTANCES = 1                                   # Number of embedding workers
EMBED_WORKER_BATCH_SIZE = 10                        # Batch size per embedding worker
EMBED_TIMEOUT_S = 1000                              # Safety timeout for embedding requests

# Async vLLM engine parameters -------------------------------------------------------
ASYNC_MODEL_NAME = "/workspace/qwen7b"
ASYNC_TENSOR_PARALLEL = 2
ASYNC_PIPELINE_PARALLEL = 1
ASYNC_DATA_PARALLEL = 2
OMP_NUM_THREADS = 4

# Sentence retrieval -----------------------------------------------------------------
K_KEY = 100                                         # Top-K keys stored per query when assigning

# train / evaluation -----------------------------------------------------------------
EXP_DIR = WORKING_DIR / "exp2"                      # Reward model training experiment folder
MODEL_SAVE_DIR = EXP_DIR / "model1"                # Output directory for trained model checkpoints

# ---------------------------------------------------------------------------
# Helpers shared across commands
# ---------------------------------------------------------------------------

engine = None
_tokenizer = None


def setup_async_engine() -> Tuple[Any, Any]:
    """Instantiate the async vLLM engine exactly as done in the notebook."""
    global engine, _tokenizer
    if engine is not None and _tokenizer is not None:
        return engine, _tokenizer

    os.environ.setdefault("OMP_NUM_THREADS", str(OMP_NUM_THREADS))

    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=ASYNC_MODEL_NAME,
        tensor_parallel_size=ASYNC_TENSOR_PARALLEL,
        pipeline_parallel_size=ASYNC_PIPELINE_PARALLEL,
        data_parallel_size=ASYNC_DATA_PARALLEL,
        gpu_memory_utilization=0.95,
        disable_log_stats=True,
    )
    local_engine = AsyncLLMEngine.from_engine_args(engine_args)
    local_engine.log_requests = False
    tokenizer = AutoTokenizer.from_pretrained(ASYNC_MODEL_NAME, padding_side="left")

    engine = local_engine
    _tokenizer = tokenizer
    return engine, tokenizer


class AllRequests:
    """Direct port of the notebook helper that fans prompts out to vLLM."""

    def __init__(self, max_request: int):
        self.max_request = max_request
        self.requests: List[Dict[str, Any]] = []
        self.request_ids: List[int] = []
        self.request_id = 0
        self.results: List[Dict[str, Any]] = []
        self.finished_ids: List[int] = []
        self.progress_bar = None

    def add(self, request: Dict[str, Any]) -> None:
        self.requests.append(request)
        self.request_ids.append(self.request_id)
        self.request_id += 1

    async def process(
        self,
        *,
        model: Optional[str] = None,
        max_tokens: int = 3000,
        temperature: float = 0.4,
        save_dir: str = "progress_log",
        restart: bool = False,
    ) -> List[Dict[str, Any]]:
        import json
        import os
        from tqdm import tqdm
        from vllm import SamplingParams

        engine, _ = setup_async_engine()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if restart:
            finished_path = Path(save_dir) / "finished_ids.json"
            results_path = Path(save_dir) / "results.json"
            if finished_path.exists() and results_path.exists():
                with finished_path.open() as f:
                    finished_ids = json.load(f)
                with results_path.open() as f:
                    self.results = json.load(f)
                for finished_id in finished_ids:
                    if finished_id in self.request_ids:
                        idx = self.request_ids.index(finished_id)
                        self.request_ids.pop(idx)
                        self.requests.pop(idx)
                        self.finished_ids.append(finished_id)
        else:
            for filename in ("finished_ids.json", "results.json"):
                path = Path(save_dir) / filename
                if path.exists():
                    path.unlink()

        total = len(self.requests)
        self.progress_bar = tqdm(total=total, desc="Processing Requests")

        async def worker() -> None:
            nonlocal engine
            while self.requests:
                request_dict = self.requests.pop(0)
                request_id = self.request_ids.pop(0)
                prompt = request_dict["prompt"]

                final_output = None
                generator = engine.generate(
                    prompt,
                    SamplingParams(temperature=temperature, max_tokens=max_tokens),
                    request_id,
                )
                async for request_output in generator:
                    final_output = request_output

                output = final_output.outputs[0].text if final_output else ""
                request_dict["output"] = output

                try:
                    json.dumps(request_dict)
                except Exception as exc:
                    print("Request serialisation error", request_id, exc)
                else:
                    self.results.append(request_dict)
                    self.finished_ids.append(request_id)
                    with (Path(save_dir) / "results.json").open("w") as f:
                        json.dump(self.results, f)
                    with (Path(save_dir) / "finished_ids.json").open("w") as f:
                        json.dump(self.finished_ids, f)

                self.progress_bar.update(1)

        await asyncio.gather(*[worker() for _ in range(max(self.max_request, 1))])
        self.progress_bar.close()
        return self.results


def extract_text(text_c: str, text_b: str) -> Optional[str]:
    import re

    pattern = rf"<{re.escape(text_b)}>(.*?)</{re.escape(text_b)}>"
    match = re.search(pattern, text_c, flags=re.DOTALL)
    return match.group(1) if match else None


def extract_int(text: str) -> Optional[int]:
    import re

    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None


def datasetdict_to_pandas(dataset_dict) -> "pd.DataFrame":  # type: ignore[name-defined]
    import pandas as pd

    frames = []
    for split_name, split_ds in dataset_dict.items():
        df = split_ds.to_pandas()
        df["split"] = split_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Notebook command ports
# ---------------------------------------------------------------------------

def process_data() -> None:
    """Port of the "Process Data" section."""
    from datasets import DatasetDict, Features, Value, load_dataset
    import pandas as pd

    dataset_name = DATASET_NAME
    save_name = DATA_SAVE_NAME

    output_dir = DATA_OUTPUT_ROOT / save_name
    output_dir.mkdir(parents=True, exist_ok=True)

    features = Features({
        "index": Value("int64"),
        "text": Value("string"),
    })

    ds = load_dataset(
        dataset_name,
        name=DATASET_CONFIG,
        split=DATA_SPLIT,
        features=features,
    )
    if "index" in ds.column_names:
        ds = ds.remove_columns("index")

    ds = ds.shuffle(seed=RANDOM_SEED)

    train_ds = ds.select(range(min(N_SAMPLE_TRAIN, len(ds))))
    if N_SAMPLE_TEST > 0:
        test_ds = ds.select(range(min(N_SAMPLE_TEST, len(ds))))
    else:
        test_ds = None

    dataset_dict = DatasetDict({"train": train_ds})
    if test_ds is not None:
        dataset_dict["test"] = test_ds

    df_all = datasetdict_to_pandas(dataset_dict)

    dataset_dict.save_to_disk(str(output_dir))
    df_all.to_csv(output_dir / "df.csv", index=False)

    if N_SMALL_SAMPLE > 0:
        df_small = df_all.sample(min(N_SMALL_SAMPLE, len(df_all)), random_state=RANDOM_SEED)
        df_small.to_csv(output_dir / "df_small.csv", index=False)

    print("Saved processed dataset to", output_dir)


def cmd_process_data(_: argparse.Namespace) -> None:
    process_data()


def generate_tags() -> None:
    """Port of the "Generate Tag Graph2" section (tag generation step)."""
    import json
    import pandas as pd
    import random

    from generate_tag_graph2 import generate_tag

    data_dir = WORKING_DIR / "data" / DATA_SAVE_NAME
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_name = "df_small.csv"  # Important: adjust if using a different sample file
    df = pd.read_csv(DATA_OUTPUT_ROOT / DATA_SAVE_NAME / csv_name)
    keys = df["text"].to_list()

    random.seed(0)
    print(">>> Generating tags with vLLM ...")
    tag_recs = generate_tag(keys, model_name=GEN_MODEL, batch_size=1000)
    for record in tag_recs[:2]:
        print(f"[key_id={record['key_id']}] {record['key']}\n  tags={record['tags']}")

    with (data_dir / "tag_recs.json").open("w") as f:
        json.dump(tag_recs, f)

    print("Saved tag records to", data_dir / "tag_recs.json")


def cmd_generate_tags(_: argparse.Namespace) -> None:
    generate_tags()


def embed_tags_step() -> None:
    """Port of embedding step from "Generate Tag Graph2"."""
    import json
    import torch

    from generate_tag_graph2 import _embed_pool_context, embed_tags as embed_tags_fn

    data_dir = WORKING_DIR / "data" / DATA_SAVE_NAME
    tag_recs_path = data_dir / "tag_recs.json"
    if not tag_recs_path.exists():
        raise FileNotFoundError(f"Missing tag records at {tag_recs_path}")

    with tag_recs_path.open() as f:
        tag_recs = json.load(f)

    pool_settings = {
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "num_instances": NUM_INSTANCES,
        "device_groups": None,
    }

    with _embed_pool_context(EMB_MODEL, **pool_settings) as pool:
        emb = embed_tags_fn(
            tag_records=tag_recs,
            embed_model_name=EMB_MODEL,
            pool=pool,
            worker_batch_size=EMBED_WORKER_BATCH_SIZE,
            timeout_s=EMBED_TIMEOUT_S,
        )

    print("embeddings:", tuple(emb.shape), emb.dtype, emb.device)
    torch.save(emb, data_dir / "key_embeddings.pt")
    print("Saved embeddings to", data_dir / "key_embeddings.pt")


def cmd_embed_tags(_: argparse.Namespace) -> None:
    embed_tags_step()


def make_tag_tree() -> None:
    """Port of grouping/tree creation from "Generate Tag Graph2"."""
    import json
    import torch

    from generate_tag_graph2 import (
        generate_representative_tag,
        generate_tag_tree as generate_tag_tree_fn,
        get_tag_group,
    )

    data_dir = WORKING_DIR / "data" / DATA_SAVE_NAME

    with (data_dir / "tag_recs.json").open() as f:
        tag_recs = json.load(f)
    emb = torch.load(data_dir / "key_embeddings.pt")

    print("\n>>> Clustering keys into groups ...")
    tag_recs, centroids, group_recs = get_tag_group(
        tag_records=tag_recs,
        embeddings=emb,
        n_group=N_GROUP,
    )
    torch.save(centroids, data_dir / "centroids.pt")
    with (data_dir / "tag_recs.json").open("w") as f:
        json.dump(tag_recs, f)
    with (data_dir / "group_recs.json").open("w") as f:
        json.dump(group_recs, f)

    print("\n>>> Generating representative tag per group with vLLM ...")
    group_recs = generate_representative_tag(
        tag_records=tag_recs,
        group_records=group_recs,
        n_tag_sample=N_TAG_SAMPLE,
        model_name=GEN_MODEL,
    )
    with (data_dir / "group_recs.json").open("w") as f:
        json.dump(group_recs, f)

    centroids = torch.load(data_dir / "centroids.pt")
    tag_tree_recs = generate_tag_tree_fn(
        group_records=group_recs,
        centroids=centroids,
        tree_struc=[500, 50, 10],
        n_tag_sample=20,
        model_name=GEN_MODEL,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        num_instances=NUM_INSTANCES,
        worker_batch_size=100,
    )
    with (data_dir / "tag_tree_recs.json").open("w") as f:
        json.dump(tag_tree_recs, f)

    print("Saved tree records to", data_dir / "tag_tree_recs.json")


def cmd_make_tag_tree(_: argparse.Namespace) -> None:
    make_tag_tree()



def get_node_by_path(tag_tree_recs: List[Dict[str, Any]], tag_ids: List[int]) -> Dict[str, Any]:
    node: Dict[str, Any] = {"children": tag_tree_recs}
    for idx in tag_ids:
        node = node["children"][idx]
    return node


def set_representative_tag(
    tag_tree_recs: List[Dict[str, Any]],
    tag_ids: List[int],
    representative_tag: str,
) -> List[Dict[str, Any]]:
    if not tag_ids:
        raise ValueError("tag_ids should include at least one index inside the list")
    node = get_node_by_path(tag_tree_recs, tag_ids)
    node["tag"] = representative_tag
    return tag_tree_recs


def is_leaf(node: Dict[str, Any]) -> bool:
    return "children" not in node or not node["children"]


async def get_representative_tag_request(
    tag_tree_recs: List[Dict[str, Any]],
    n_tag_sample: int = N_TAG_SAMPLE,
) -> Tuple[List[Dict[str, Any]], bool]:
    all_requests = AllRequests(max_request=10)
    progress_made = False

    def ensure_leaf_tags(node: Dict[str, Any]) -> None:
        nonlocal progress_made
        if is_leaf(node):
            if "tag" not in node:
                tags = node.get("tags", [])
                node["tag"] = tags[0] if tags else "general"
                progress_made = True
        else:
            for child in node["children"]:
                ensure_leaf_tags(child)

    def enqueue_when_children_tagged(node_list: List[Dict[str, Any]], path: List[int]) -> None:
        for idx, child in enumerate(node_list):
            if not is_leaf(child):
                enqueue_when_children_tagged(child["children"], path + [idx])

        if path == []:
            return

        parent_node = get_node_by_path(tag_tree_recs, path)
        if "tag" in parent_node:
            return

        child_tags = [child.get("tag") for child in node_list if "tag" in child]
        if len(child_tags) != len(node_list):
            return

        sample = (
            child_tags
            if len(child_tags) <= n_tag_sample
            else random.sample(child_tags, n_tag_sample)
        )
        if not sample:
            sample = ["general"]
        lines = "\n".join(f"- {t}" for t in sample)
        prompt = (
            "You are a taxonomy expert. Given the following sample tags from one cluster,\n"
            "produce ONE concise representative tag (≤ 6 words) that best describes them all.\n"
            "Do NOT include punctuation at the end. Output ONLY the tag text, nothing else.\n\n"
            f"Sample tags:\n{lines}\n\nRepresentative tag:"
        )
        all_requests.add({"tag_ids": path, "prompt": prompt})

    synth_root = {"children": tag_tree_recs}
    ensure_leaf_tags(synth_root)
    enqueue_when_children_tagged(tag_tree_recs, [])

    if not all_requests.requests and not progress_made:
        return tag_tree_recs, True

    results = await all_requests.process(max_tokens=3000, temperature=0.0, save_dir="progress_questions", restart=False)
    for result in results:
        tag_ids = result["tag_ids"]
        output = result["output"]
        set_representative_tag(tag_tree_recs, tag_ids, output)

    return tag_tree_recs, False


async def build_representative_tags(tag_tree_recs: List[Dict[str, Any]], save_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    while_end = False
    while not while_end:
        tag_tree_recs, while_end = await get_representative_tag_request(tag_tree_recs)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(tag_tree_recs, f, ensure_ascii=False, indent=2)

    return tag_tree_recs


def make_parent_tags() -> None:
    """Async parent-tag labelling from the notebook."""
    data_dir = WORKING_DIR / "data" / DATA_SAVE_NAME
    tag_tree_path = data_dir / "tag_tree_recs.json"
    if not tag_tree_path.exists():
        raise FileNotFoundError(f"Missing tag tree records at {tag_tree_path}")

    with tag_tree_path.open() as f:
        tag_tree_recs = json.load(f)

    save_path = data_dir / "tag_tree_recs.json"
    asyncio.run(build_representative_tags(tag_tree_recs, save_path))
    print("Saved updated tree with parent tags to", save_path)


def cmd_make_parent_tags(_: argparse.Namespace) -> None:
    make_parent_tags()


def make_queries() -> None:
    """Port of the "Make queries" section."""
    import pandas as pd
    import re

    _, tokenizer = setup_async_engine()

    save_name = DATA_SAVE_NAME
    csv_name = "df_small.csv"
    df = pd.read_csv(DATA_OUTPUT_ROOT / save_name / csv_name)

    all_requests = AllRequests(max_request=50)

    for request_id in range(len(df)):
        system_prompt = (
            "You are a helpful assistant who extracts titles, and keywords from a sentence provided by the user,"
            " and also creates questions and irrelevant questions.\n"
            "Following the user's instructions, analyze the content of the sentence and respond according to the output format below.\n"
            "Make sure that your questions are creative and sometimes that asks question\n\n"
            "Output format:\n"
            "<titles>[\"Title1\", \"Title2\", ... ]</titles>\n"
            "<keywords>[\"Keyword1\", \"Keyword2\", ... ]</keywords>\n"
            "<questions>[\"Question1\", \"Question2\", ... ]</questions>\n"
            "<irrelevant questions>[\"Question1\", \"Question2\", ... ]</irrelevant questions>"
        )
        user_prompt = (
            f"Sentence:\n'''\n{df.iloc[request_id]['text']}\n'''\n\n"
            "Instructions:\n"
            "1. Summarize the content of the sentence into 2-3 one-line titles.\n"
            "2. Extract 3–5 main keywords from the sentence.\n"
            "3. Create several questions and irrelevant ones about the sentence, ranging from easy to difficult.\n"
            "4. Enclose each element in order with the tags <titles></titles>, <keywords></keywords>,"
            " and <irrelevant questions></irrelevant questions> when outputting.\n\n"
            "Follow the instructions step-by-step and think in sequence."
        )

        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        request = {"request_id": request_id, "prompt": prompt}
        all_requests.add(request)

    results = asyncio.run(
        all_requests.process(max_tokens=3000, temperature=0.0, save_dir="progress_questions", restart=True)
    )

    query_dict: Dict[int, Dict[str, List[str]]] = {}
    for result in results:
        request_id = result["request_id"]
        output = result["output"]
        titles = extract_text(output, "titles")
        keywords = extract_text(output, "keywords")
        questions = extract_text(output, "questions")
        irr_questions = extract_text(output, "irrelevant questions")

        try:
            titles_list = json.loads(titles) if titles else []
            keywords_list = json.loads(keywords) if keywords else []
            questions_list = json.loads(questions) if questions else []
            irr_questions_list = json.loads(irr_questions) if irr_questions else []
        except Exception:
            titles_list = []
            keywords_list = []
            questions_list = []
            irr_questions_list = []

        query_dict[request_id] = {
            "titles": titles_list,
            "keywords": keywords_list,
            "questions": questions_list,
            "irr_questions": irr_questions_list,
        }

    data_dir = DATA_OUTPUT_ROOT / save_name
    with (data_dir / "query_dict.json").open("w") as f:
        json.dump(query_dict, f)
    print("Saved query_dict to", data_dir / "query_dict.json")


def cmd_make_queries(_: argparse.Namespace) -> None:
    make_queries()


def rm_assign_keys() -> None:
    """Port of the "Reward Model Gets TopN-Relevant Sentences..." section."""
    import asyncio
    import json
    import multiprocessing as mp
    import os
    import random
    from copy import deepcopy
    from pathlib import Path

    import pandas as pd
    import torch
    from vllm_reward2 import build_llm, search

    save_name = DATA_SAVE_NAME
    local_data_dir = DATA_OUTPUT_ROOT / save_name
    workspace_data_dir = WORKING_DIR / "data" / save_name
    workspace_data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(local_data_dir / "df_small.csv")
    with (local_data_dir / "query_dict.json").open() as f:
        _query_dict_json = json.load(f)  # Loaded to mirror notebook workflow
    with (workspace_data_dir / "tag_tree_recs.json").open() as f:
        tag_dict = json.load(f)

    tensor_parallel_size = 1  # Important: number of tensor parallel shards for the RM model
    num_instances = 4  # Important: number of RM worker replicas to launch

    mp.set_start_method("spawn", force=True)
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "1")

    device_groups: list[list[int]] = []
    device_id = 0
    for _ in range(num_instances):
        group: list[int] = []
        for _ in range(tensor_parallel_size):
            group.append(device_id)
            device_id += 1
        device_groups.append(group)

    rm = build_llm(
        model_name="/workspace/llama3b-rm-converted-model",
        tensor_parallel_size=len(device_groups[0]),
        num_instances=len(device_groups),
        device_groups=device_groups,
        max_model_len=2500,
        max_num_seqs=64,
        gpu_memory_utilization=0.90,
        runner="pooling",
    )
    tokenizer = rm.tokenizer

    def llm_template_func(row: dict[str, str]) -> str:
        message = [
            {"role": "user", "content": f"Generate tag for the sentence\n\nSentence:'''{row['query']}'''"},
            {"role": "assistant", "content": f"{row['key']}"},
        ]
        if len(message[0]["content"]) > 4000:
            message[0]["content"] = message[0]["content"][:4000] + "..."
        return tokenizer.apply_chat_template(message, tokenize=False)

    cache_dir = workspace_data_dir / "rm_assign_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def get_tag_dict(tag_ids: list[int], tree: list[dict[str, object]]):
        if not tag_ids:
            return None
        node = tree
        for tag_id in tag_ids[:-1]:
            if "children" not in node[tag_id]:
                return None
            node = node[tag_id]["children"]
        return node[tag_ids[-1]]

    def get_tag(tag_ids: list[int], tree: list[dict[str, object]]):
        if not tag_ids:
            return None
        node = tree
        for tag_id in tag_ids[:-1]:
            node = node[tag_id]["children"]
        return node[tag_ids[-1]]["tag"]

    def set_key_id(tag2key: list[dict[str, object]], tag_ids: list[int], key_id: int):
        subtree = tag2key
        for i, tag_id in enumerate(tag_ids):
            tag_dict = subtree[tag_id]
            tag_dict.setdefault("key_ids", []).append(key_id)
            if len(tag_ids) > i + 1 and "children" in tag_dict:
                subtree = tag_dict["children"]
        return tag2key

    def set_query_id(tag2query: list[dict[str, object]], tag_ids: list[int], query_id: int):
        subtree = tag2query
        for tag_id in tag_ids:
            tag_dict = subtree[tag_id]
            tag_dict.setdefault("query_ids", []).append(query_id)
            if "children" in tag_dict:
                subtree = tag_dict["children"]
        return tag2query

    async def search_tag(query_dict: list[dict[str, str]], tree: list[dict[str, object]], k_tag: int = 2):
        tag2query = deepcopy(tree)
        query2tag_ids = [{"tag_ids": [[] for _ in range(k_tag)]} for _ in range(len(query_dict))]

        tags = [tag_node["tag"] for tag_node in tree]
        tags_request = [{"tags": [tags]} for _ in range(len(query_dict))]
        depth = 1
        while_end = False

        while not while_end:
            requests: list[dict[str, object]] = []
            query_and_n_top_ids: list[tuple[int, int]] = []
            total_requests = 0

            for query_id, tag_record in enumerate(tags_request):
                for nth_tag, tag_list in enumerate(tag_record["tags"]):
                    query_and_n_top_ids.append((query_id, nth_tag))
                    requests.append({"query": query_dict[query_id]["query"], "keys": tag_list, "k": k_tag, "return_relevance": True})
                    total_requests += len(tag_list)

            cache_file = cache_dir / f"output{depth}.json"
            if cache_file.exists():
                with cache_file.open() as f:
                    output = json.load(f)
            else:
                batch_size = max(total_requests // num_instances, 1)
                print(f"Graph Depth: {depth},  total_requests: {total_requests},  Batch size: {batch_size}")
                output = search(rm, requests, llm_template_func, topk=k_tag, batch_size=1000, timeout_s=10000)
                with cache_file.open("w") as f:
                    json.dump(output, f)

            tags_request = [{"tags": []} for _ in range(len(query_dict))]
            result_holder = {idx: {"tag_ids_list": [], "relevance_list": []} for idx in range(len(query_dict))}

            for request_index, output_dict in enumerate(output):
                query_id, nth_tag_ids = query_and_n_top_ids[request_index]
                prior_tag_ids = query2tag_ids[query_id]["tag_ids"][nth_tag_ids]
                for top_nth in range(k_tag):
                    try:
                        new_tag_id = output_dict["keys"][top_nth]["key_id"]
                        relevance = output_dict["keys"][top_nth]["relevance"]
                        result_holder[query_id]["tag_ids_list"].append(prior_tag_ids + [new_tag_id])
                        result_holder[query_id]["relevance_list"].append(relevance)
                    except Exception:
                        continue

            while_end = True

            for query_id, holder in result_holder.items():
                tag_relevance = holder["relevance_list"]
                tag_ids_list = holder["tag_ids_list"]
                if not tag_relevance:
                    top_tag_ids_list: list[list[int]] = []
                elif len(tag_relevance) < k_tag:
                    _, indices = torch.topk(torch.tensor(tag_relevance), k=len(tag_relevance))
                    top_tag_ids_list = [tag_ids_list[index.item()] for index in indices]
                else:
                    _, indices = torch.topk(torch.tensor(tag_relevance), k=k_tag)
                    top_tag_ids_list = [tag_ids_list[index.item()] for index in indices]

                query2tag_ids[query_id]["tag_ids"] = top_tag_ids_list
                for tag_ids in top_tag_ids_list:
                    tag_info = get_tag_dict(tag_ids, tag2query)
                    if not tag_info or "children" not in tag_info or tag_info["children"] == []:
                        continue
                    child_tags = [child["tag"] for child in tag_info["children"]]
                    while_end = False
                    tags_request[query_id]["tags"].append(child_tags)

            depth += 1

        for query_id in range(len(query_dict)):
            for tag_ids in query2tag_ids[query_id]["tag_ids"]:
                set_query_id(tag2query, tag_ids, query_id)

        return query2tag_ids, tag2query

    key_dict = [{"query": df.iloc[i]["text"]} for i in range(len(df))]

    query2tag_ids, tag2query = asyncio.run(search_tag(key_dict, tag_dict))

    with (workspace_data_dir / "query2tag_ids-tag_tree.json").open("w") as f:
        json.dump(query2tag_ids, f)
    with (workspace_data_dir / "tag2query-tag_tree.json").open("w") as f:
        json.dump(tag2query, f)

    print("All Finished")

def cmd_rm_assign_keys(_: argparse.Namespace) -> None:
    rm_assign_keys()


def judge_sentence() -> None:
    """Port of the pairwise judgement section."""
    import itertools
    import pandas as pd

    _, tokenizer = setup_async_engine()

    save_name = DATA_SAVE_NAME
    data_dir = DATA_OUTPUT_ROOT / save_name

    df = pd.read_csv(data_dir / "df_small.csv")
    with (data_dir / "query_dict.json").open() as f:
        query_dict = json.load(f)
    with (data_dir / "sentences_relevant_to_questions.json").open() as f:
        relevant_sentences = json.load(f)

    print("len(relevant_sentences):", len(relevant_sentences))

    all_requests = AllRequests(max_request=40)

    for sentence_dict in relevant_sentences:
        query_id = sentence_dict["query_id"]
        correct_id = sentence_dict["correct_id"]
        query = sentence_dict["query"]
        sentence_ids = [key_dict["key_id"] for key_dict in sentence_dict["keys"]]
        sentences = [key_dict["key"] for key_dict in sentence_dict["keys"]]

        sentence_ids = sentence_ids[:3]
        if correct_id not in sentence_ids[:2]:
            sentence_ids = sentence_ids[:2] + [int(correct_id)]

        sentence_id_pairs = list(itertools.combinations(sentence_ids, 2))
        if not sentence_id_pairs:
            continue
        sample_sentence_id_pairs = random.sample(sentence_id_pairs, 1)

        for request_id, pair in enumerate(sample_sentence_id_pairs):
            sentence_id1, sentence_id2 = pair
            system_prompt = (
                "You are a brilliant judge who decides which text is more relevant to a given query.\n"
                "You will be given a query, 2 sentences.\n"
                "Please carefully analyze these two sentences and then return your answer following the output format.\n\n"
                "Output format:\n<ID> 1 or 2 (file id more relevant to given query) </ID>"
            )
            user_prompt = (
                f"Query: {query}\n\n"
                f"Sentence 1:\n'''{df.iloc[sentence_id1]['text']}'''\n\n"
                f"Sentence 2:\n'''{df.iloc[sentence_id2]['text']}'''\n\n"
                "Instructions:\n"
                "1. Analyze content of each sentence.\n"
                "2. Think which sentence is more relevant to the query.\n"
                "3. Please return id of the more relevant sentence enclosing it within <ID></ID> tag.\n"
                "4. If both sentences are highly related to the query or either of sentence is totally irrelevant to it, return -1 in the <ID></ID> tag.\n\n"
                "Let's think step by step following each step of the instructions."
            )

            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

            request = {
                "request_id": request_id,
                "prompt": prompt,
                "sentence_ids": [sentence_id1, sentence_id2],
                "question": query,
            }
            all_requests.add(request)

    print("len(all_requests.requests):", len(all_requests.requests))

    results = asyncio.run(
        all_requests.process(max_tokens=3000, temperature=0.0, save_dir="relevant_file_progress7", restart=True)
    )
    print("Completed judgements:", len(results))


def cmd_judge_sentence(_: argparse.Namespace) -> None:
    judge_sentence()


def make_dataset_list() -> None:
    """Port of the dataset assembly section."""
    import pandas as pd

    progress_dir = Path("relevant_file_progress7")
    results_path = progress_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing judge results at {results_path}")

    with results_path.open() as f:
        results = json.load(f)

    save_name = DATA_SAVE_NAME
    data_dir = DATA_OUTPUT_ROOT / save_name

    df = pd.read_csv(data_dir / "df_small.csv")
    with (data_dir / "query_dict.json").open() as f:
        query_dict = json.load(f)
    with (data_dir / "sentences_relevant_to_questions.json").open() as f:
        relevant_sentences = json.load(f)

    dataset_list: List[Dict[str, Any]] = []

    for result in results:
        output = result["output"]
        sentence_ids = result["sentence_ids"]
        question = result["question"]
        chosen_id = extract_text(output, "ID")
        if chosen_id is None:
            chosen_id = extract_int(output[-10:])
        try:
            chosen_id_int = int(chosen_id)
        except Exception:
            continue
        if chosen_id_int not in (1, 2):
            continue

        if chosen_id_int == 1:
            chosen_sentence_id = sentence_ids[0]
            rejected_sentence_id = sentence_ids[1]
        else:
            chosen_sentence_id = sentence_ids[1]
            rejected_sentence_id = sentence_ids[0]

        dataset_list.append(
            {
                "chosen_msg": [
                    {
                        "role": "user",
                        "content": (
                            "Give me relevant score between query and sentence;\n\n"
                            f"Query:{question}\n\n"
                            f"Sentence:```{df.iloc[chosen_sentence_id]['text']}```"
                        ),
                    }
                ],
                "rejected_msg": [
                    {
                        "role": "user",
                        "content": (
                            "Give me relevant score between query and sentence;\n\n"
                            f"Query:{question}\n\n"
                            f"Sentence:```{df.iloc[rejected_sentence_id]['text']}```"
                        ),
                    }
                ],
                "chosen_sentence_id": chosen_sentence_id,
                "rejected_sentence_id": rejected_sentence_id,
            }
        )

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    dataset_list_path = EXP_DIR / "dataset_list.json"
    with dataset_list_path.open("w") as f:
        json.dump(dataset_list, f)
    print("Saved dataset_list to", dataset_list_path)


def cmd_make_dataset_list(_: argparse.Namespace) -> None:
    make_dataset_list()


def train_reward_model() -> None:
    """Port of the reward model training section."""
    import json

    from peft import LoraConfig, TaskType
    from rmsearch import RMTrainer
    from trl import RewardConfig, RewardTrainer

    model_name = "/workspace/llama3b-rm"
    num_gpus = 2

    rmtrainer = RMTrainer(model_name=model_name, num_gpus=num_gpus)
    dataset_list_path = EXP_DIR / "dataset_list.json"
    with dataset_list_path.open() as f:
        dataset_list = json.load(f)

    tokenizer = rmtrainer.tokenizer

    def formatting_func(examples):
        kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": 4000,
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        chosen_msg = examples["chosen_msg"]
        rejected_msg = examples["rejected_msg"]

        if len(chosen_msg[0]["content"]) > 4000:
            chosen_msg[0]["content"] = chosen_msg[0]["content"][:4000] + "..."
        if len(rejected_msg[0]["content"]) > 4000:
            rejected_msg[0]["content"] = rejected_msg[0]["content"][:4000] + "..."

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_msg, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_msg, tokenize=False)

        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0],
            "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0],
            "attention_mask_rejected": tokens_rejected["attention_mask"][0],
        }

    formatted_dataset = rmtrainer.prepare_dataset(
        dataset_list,
        base_dir=EXP_DIR,
        test_size=100,
        formatting_func=formatting_func,
    )

    class CustomRewardTrainer(RewardTrainer):
        _tag_names = ["trl", "reward-trainer"]

        def train(self, *args, **kwargs):
            return super().train(*args, **kwargs)

        def evaluate(self, *args, **kwargs):
            return super().evaluate(num_print_samples=1, *args, **kwargs)

    training_args = RewardConfig(
        output_dir=str(MODEL_SAVE_DIR),
        per_device_train_batch_size=3,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=40,
        eval_on_start=True,
        save_steps=20,
        logging_steps=1,
        num_train_epochs=50,
        report_to=None,
        remove_unused_columns=False,
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=[
            "k_proj",
            "q_proj",
            "o_proj",
            "v_proj",
            "down_proj",
            "gate_proj",
            "up_proj",
        ],
        layers_to_transform=[25, 26, 27],
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
    )

    rmtrainer.train(
        formatted_dataset,
        training_args=training_args,
        peft_config=peft_config,
        trainer_cls=CustomRewardTrainer,
    )


def cmd_train(_: argparse.Namespace) -> None:
    train_reward_model()


def convert_model() -> None:
    """Port of the model conversion section."""
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    base_model_name = "/workspace/llama3b-rm"
    checkpoint_path = "exp3/model1/checkpoint-40"
    model_name = "exp3-model1-step40"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left", add_eos_token=False, add_bos_token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)
    lora_model = PeftModel.from_pretrained(model, checkpoint_path)
    reward_model = lora_model.merge_and_unload()

    save_dir = Path(f"./{model_name}-converted-model")
    score_save_path = Path(f"./{model_name}-converted-score.pt")

    tokenizer.save_pretrained(save_dir)
    reward_model.save_pretrained(save_dir)
    torch.save(reward_model.score.weight.data, score_save_path)
    del reward_model

    generate_model = AutoModelForCausalLM.from_pretrained(save_dir)
    generate_model.save_pretrained(save_dir)
    del generate_model

    print("Converted model saved to", save_dir)


def cmd_convert_model(_: argparse.Namespace) -> None:
    convert_model()


def retrieval_evaluation() -> None:
    """Port of the retrieval evaluation section."""
    import logging
    import multiprocessing as mp
    import os
    import pandas as pd
    import torch

    from vllm_reward2 import build_llm, search

    working_dir = WORKING_DIR
    model_name = "/workspace/llama3b-rm-converted-model"
    exp_eval_dir = Path(f"{model_name}-eval")
    data_name = DATA_SAVE_NAME
    tensor_parallel_size = 1
    num_instances = 4
    output_path = exp_eval_dir / "relevance_dict.json"

    exp_eval_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(working_dir / "data" / data_name / "df_small.csv")
    with (working_dir / "data" / data_name / "query_dict.json").open() as f:
        query_dict = json.load(f)
    with (working_dir / "data" / data_name / "sentences_relevant_to_questions.json").open() as f:
        relevant_sentences = json.load(f)
    with (working_dir / "data" / data_name / "tag2query-tag_tree.json").open() as f:
        tag_dict = json.load(f)

    mp.set_start_method("spawn", force=True)

    device_groups: List[List[int]] = []
    device_id = 0
    for _ in range(num_instances):
        group: List[int] = []
        for _ in range(tensor_parallel_size):
            group.append(device_id)
            device_id += 1
        device_groups.append(group)

    rm = build_llm(
        model_name=model_name,
        tensor_parallel_size=len(device_groups[0]),
        num_instances=len(device_groups),
        device_groups=device_groups,
        max_model_len=2500,
        max_num_seqs=64,
        gpu_memory_utilization=0.90,
        runner="pooling",
    )
    tokenizer = rm.tokenizer

    logging.getLogger("vllm").setLevel(logging.ERROR)

    def llm_template_func(row):
        query = row["query"]
        key = row["key"]
        message = [
            {
                "role": "user",
                "content": (
                    "Give me relevance score between\n\n"
                    f"Query:{query}\n\n"
                    f"Sentence:{key}"
                ),
            }
        ]
        if len(message[0]["content"]) > 4000:
            message[0]["content"] = message[0]["content"][:4000] + "..."
        prompt = tokenizer.apply_chat_template(message, tokenize=False)
        return prompt

    def get_tag_dict(tag_ids, tag_dict):
        if tag_ids == []:
            return None
        for tag_id in tag_ids[:-1]:
            if "children" not in tag_dict[tag_id]:
                return None
            tag_dict = tag_dict[tag_id]["children"]
        return tag_dict[tag_ids[-1]]

    def search_key(queries, keys, tag2key, k_tag=2, k_key=5, num_instances=1):
        query2tag_ids = [{"tag_ids": [[] for _ in range(k_tag)]} for _ in range(len(queries))]
        tags = [tag2key_dict["tag"] for tag2key_dict in tag2key]
        tags_request = [{"tags": [tags]} for _ in range(len(queries))]
        while_end = False
        depth = 0

        while not while_end:
            depth += 1
            requests = []
            query_and_n_top_ids = []
            total_requests = 0

            for query_id in range(len(tags_request)):
                for nth_tag_ids, tag_list in enumerate(tags_request[query_id]["tags"]):
                    query_and_n_top_ids.append((query_id, nth_tag_ids))
                    requests.append({"query": queries[query_id], "keys": tag_list, "k": k_tag, "return_relevance": True})
                    total_requests += len(tag_list)

            path = f"search_key-output{depth}-test4.json"
            if not os.path.exists(path):
                batch_size = 10
                print(f"Graph Depth: {depth},  total_requests: {total_requests},  Batch size: {batch_size}")
                output = search(rm, requests, llm_template_func, topk=k_tag, batch_size=5000, timeout_s=4000)
                with open(path, "w") as f:
                    json.dump(output, f)
            else:
                with open(path) as f:
                    output = json.load(f)

            tags_request = [{"tags": []} for _ in range(len(queries))]
            result1 = {query_id: {"tag_ids_list": [], "relevance_list": []} for query_id in range(len(queries))}
            for request_id, output_dict in enumerate(output):
                query_id, nth_tag_ids = query_and_n_top_ids[request_id]
                pre_tag_ids = query2tag_ids[query_id]["tag_ids"][nth_tag_ids]

                for top_nth in range(k_tag):
                    try:
                        new_tag_id = output_dict["keys"][top_nth]["key_id"]
                        relevance = output_dict["keys"][top_nth]["relevance"]
                        result1[query_id]["tag_ids_list"].append(pre_tag_ids + [new_tag_id])
                        result1[query_id]["relevance_list"].append(relevance)
                    except Exception:
                        continue

            while_end = True

            for query_id in result1:
                tag_relevance = result1[query_id]["relevance_list"]
                tag_ids_list = result1[query_id]["tag_ids_list"]
                if len(tag_relevance) == 0:
                    top_tag_ids_list = []
                elif len(tag_relevance) < k_tag:
                    _, indices = torch.topk(torch.tensor(tag_relevance), k=len(tag_relevance))
                    top_tag_ids_list = [tag_ids_list[index.item()] for index in indices]
                else:
                    _, indices = torch.topk(torch.tensor(tag_relevance), k=k_tag)
                    top_tag_ids_list = [tag_ids_list[index.item()] for index in indices]

                query2tag_ids[query_id]["tag_ids"] = top_tag_ids_list
                for tag_ids in top_tag_ids_list:
                    tag_info = get_tag_dict(tag_ids, tag2key)
                    if not tag_info or "children" not in tag_info or tag_info["children"] == []:
                        continue
                    tags = [child["tag"] for child in tag_info["children"]]
                    while_end = False
                    tags_request[query_id]["tags"].append(tags)

        query2key_ids = []
        requests = []
        total_requests = 0
        for query_id, record in enumerate(query2tag_ids):
            combined_key_ids = []
            for tag_ids in record["tag_ids"]:
                tag_info = get_tag_dict(tag_ids, tag2key)
                if "query_ids" in tag_info:
                    combined_key_ids += tag_info["query_ids"]
            query2key_ids.append({"key_ids": combined_key_ids})
            selected_keys = []
            for key_id in combined_key_ids:
                if key_id < len(keys):
                    selected_keys.append(keys[key_id])
            requests.append({"query": queries[query_id], "keys": selected_keys, "k": k_key, "return_relevance": True})
            total_requests += len(selected_keys)

        batch_size = 10
        print(f"Final Search,  total_requests: {total_requests},  Batch size: {batch_size}")
        output = search(rm, requests, llm_template_func, topk=k_key, batch_size=batch_size, timeout_s=10000)
        return output

    sentences = [df.iloc[i]["text"] for i in range(len(df))]

    questions = []
    correct_ids = []
    for idx in range(len(df)):
        q = query_dict[str(idx)]["questions"]
        questions += q
        correct_ids += [idx for _ in range(len(q))]

    output = search_key(questions, sentences, tag_dict, k_tag=2, k_key=10, num_instances=num_instances)

    for i in range(len(output)):
        output[i]["correct_id"] = correct_ids[i]
        for j in range(len(output[i]["keys"])):
            output[i]["keys"][j]["relevant_id"] = output[i]["keys"][j]["key_id"]

    with output_path.open("w") as f:
        json.dump(output, f)
    print("file_saved to", output_path)


def cmd_retrieval_evaluation(_: argparse.Namespace) -> None:
    retrieval_evaluation()


COMMANDS: Dict[str, Any] = {
    "process_data": cmd_process_data,
    "generate_tags": cmd_generate_tags,
    "embed_tags": cmd_embed_tags,
    "make_tag_tree": cmd_make_tag_tree,
    "make_parent_tags": cmd_make_parent_tags,
    "make_queries": cmd_make_queries,
    "rm_assign_keys": cmd_rm_assign_keys,
    "judge_sentence": cmd_judge_sentence,
    "make_dataset_list": cmd_make_dataset_list,
    "train": cmd_train,
    "convert_model": cmd_convert_model,
    "retrieval_evaluation": cmd_retrieval_evaluation,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run individual steps from train_en.ipynb")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparsers.add_parser(command, help=f"Run {command} step")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMANDS[args.command]
    handler(args)


if __name__ == "__main__":
    main()
