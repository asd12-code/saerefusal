#!/usr/bin/env python3
"""Run reproducible steering sweeps with SAE and DiM directions."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from datasets import load_dataset
from sae_lens import SAE
from tqdm import tqdm

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base


SAE_CONFIG = {
    "google/gemma-3-1b-it": {
        "release": "gemma-scope-2-1b-it-res-all",
        "sae_id": "layer_16_width_16k_l0_big",
        "target_layer": 16,
        "feature_idx": 3968,
        "cos_sim": 0.496,
    },
    "google/gemma-3-4b-it": {
        "release": "gemma-scope-2-4b-it-res-all",
        "sae_id": "layer_16_width_16k_l0_big",
        "target_layer": 16,
        "feature_idx": 1648,
        "cos_sim": 0.639,
    },
    "google/gemma-3-12b-it": {
        "release": "gemma-scope-2-12b-it-res-all",
        "sae_id": "layer_28_width_262k_l0_big",
        "target_layer": 28,
        "feature_idx": 1118,
        "cos_sim": 0.594,
    },
    "google/gemma-3-27b-it": {
        "release": "gemma-scope-2-27b-it-res-all",
        "sae_id": "layer_39_width_262k_l0_big",
        "target_layer": 39,
        "feature_idx": 6107,
        "cos_sim": 0.585,
    },
}

DEFAULT_SEED = 42
DEFAULT_FACTORS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
RANKED_FEATURE_TOPK = 5

HARMFUL_TEST_PATH = Path("steering_dataset/harmful_test.json")

STEERING_DATASET_MAP = {
    "alpaca_eval": Path(
        "steering_dataset/alpaca_eval.json"
    ),
    "rewrite_alpaca": Path(
        "steering_dataset/rewrite_data.json"
    ),
}

ACTIVATION_STEERING_MAP = {
    "google/gemma-3-1b-it": Path(
        "gemma-3-1b-it_topkactivation_layer16.json"
    ),
    "google/gemma-3-4b-it": Path(
        "gemma-3-4b-it_topkactivation_layer16.json"
    ),
    "google/gemma-3-12b-it": Path(
        "gemma-3-12b-it_topkactivation_layer28.json"
    ),
    "google/gemma-3-27b-it": Path(
        "gemma-3-27b-it_topkactivation_layer39.json"
    ),
}

DIM_VECTOR_MAP = {
    "google/gemma-3-1b-it": Path(
        "pipeline/runs/gemma-3-1b-it/direction.pt"
    ),
    "google/gemma-3-4b-it": Path(
        "pipeline/runs/gemma-3-4b-it/gemma-3-4b-it_direction.pt"
    ),
    "google/gemma-3-12b-it": Path(
        "pipeline/runs/gemma-3-12b-it/direction.pt"
    ),
    "google/gemma-3-27b-it": Path(
        "pipeline/runs/gemma-3-27b-it/direction.pt"
    ),
}

DIM_VECTOR_METADATA_MAP = {
    "google/gemma-3-1b-it": Path(
        "pipeline/runs/gemma-3-1b-it/direction_metadata.json"
    ),
    "google/gemma-3-4b-it": Path(
        "pipeline/runs/gemma-3-4b-it/gemma-3-4b-it_direction_metadata.json"
    ),
    "google/gemma-3-12b-it": Path(
        "pipeline/runs/gemma-3-12b-it/direction_metadata.json"
    ),
    "google/gemma-3-27b-it": Path(
        "pipeline/runs/gemma-3-27b-it/direction_metadata.json"
    ),
}


@dataclass(frozen=True)
class EvaluationSet:
    name: str
    split: str
    analysis_name: str
    prompts: list[str]


@dataclass
class SteeringRun:
    method: str
    direction: torch.Tensor
    layer: int
    sae: Any = None
    sae_id: str | None = None
    feature_idx: int | None = None
    target_feature_idx: int | None = None
    sample_id: int | None = None
    random_seed: int | None = None
    using_dim_vector: bool = False


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def format_prompt(user_prompt: str) -> str:
    return (
        "<start_of_turn>user\n"
        f"{user_prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {path}, got {type(data).__name__}."
        )

    return data


def load_evaluation_set(
    dataset_name: str,
    use_safe: bool,
) -> EvaluationSet:
    split = "safe" if use_safe else "unsafe"

    if dataset_name == "xstest":
        dataset = load_dataset("walledai/XSTest", split="test")
        label = "safe" if use_safe else "unsafe"
        subset = dataset.filter(lambda row: row["label"] == label)
        prompts = [str(row["prompt"]) for row in subset]
        analysis_name = f"xstest_{split}"

    elif dataset_name == "wildguardmix":
        dataset = load_dataset(
            "allenai/wildguardmix",
            "wildguardtest",
            split="test",
        )
        dataset = dataset.filter(
            lambda row: row["prompt_harm_label"] is not None
        )
        label = "unharmful" if use_safe else "harmful"
        subset = dataset.filter(
            lambda row: row["prompt_harm_label"] == label
        )
        prompts = [str(row["prompt"]) for row in subset]
        analysis_name = f"wildguard_{split}"

    elif dataset_name in {"alpaca_eval", "rewrite_alpaca"}:
        path = (
            STEERING_DATASET_MAP[dataset_name]
            if use_safe
            else HARMFUL_TEST_PATH
        )
        records = load_json_list(path)
        prompts = [str(row["instruction"]) for row in records]
        analysis_name = "alpaca_safe" if use_safe else "mix_unsafe"

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not prompts:
        raise ValueError(f"No prompts found for {dataset_name}/{split}.")

    print(f"Loaded {len(prompts)} prompts from {dataset_name}/{split}.")

    return EvaluationSet(
        name=dataset_name,
        split=split,
        analysis_name=analysis_name,
        prompts=prompts,
    )


def get_model_layer(model: Any, layer: int) -> Any:
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers[layer]

    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "layers")
    ):
        return model.language_model.layers[layer]

    if (
        hasattr(model, "model")
        and hasattr(model.model, "layers")
    ):
        return model.model.layers[layer]

    raise ValueError(
        "Cannot locate transformer layers on model type "
        f"{type(model).__name__}."
    )


def extract_responses(
    tokenizer: Any,
    generated_ids: torch.Tensor,
) -> list[str]:
    decoded = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )
    marker = "<start_of_turn>model"

    return [
        text.rsplit(marker, maxsplit=1)[-1].strip()
        if marker in text
        else text.strip()
        for text in decoded
    ]


@torch.no_grad()
def generate_batch(
    model: Any,
    tokenizer: Any,
    tokenized_inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    layer: int | None = None,
    direction: torch.Tensor | None = None,
    coefficient: float = 0.0,
    using_dim_vector: bool = False,
) -> list[str]:
    handle = None

    if direction is not None and coefficient != 0.0:
        target_module = get_model_layer(model, int(layer))

        def steering_hook(
            module: Any,
            inputs: Any,
            outputs: Any,
        ) -> Any:
            del module, inputs

            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            hidden = hidden.clone()

            direction_local = direction.to(
                device=hidden.device,
                dtype=hidden.dtype,
            ).reshape(1, 1, -1)

            if hidden.shape[1] == 1:
                scale: float | torch.Tensor = (
                    1.0
                    if using_dim_vector
                    else hidden.norm(dim=-1, keepdim=True)
                )
                hidden = hidden + coefficient * scale * direction_local
            else:
                scale = (
                    1.0
                    if using_dim_vector
                    else hidden[:, 1:, :].norm(dim=-1, keepdim=True)
                )
                hidden[:, 1:, :] = (
                    hidden[:, 1:, :]
                    + coefficient * scale * direction_local
                )

            if isinstance(outputs, tuple):
                return (hidden, *outputs[1:])

            return hidden

        handle = target_module.register_forward_hook(steering_hook)

    model_device = next(model.parameters()).device
    input_ids = tokenized_inputs["input_ids"].to(model_device)
    attention_mask = tokenized_inputs["attention_mask"].to(model_device)

    try:
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    finally:
        if handle is not None:
            handle.remove()

    return extract_responses(tokenizer, generated_ids)


def iter_batches(
    items: list[str],
    batch_size: int,
) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def tokenize_prompts(
    tokenizer: Any,
    prompts: list[str],
) -> dict[str, torch.Tensor]:
    return tokenizer(
        [format_prompt(prompt) for prompt in prompts],
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )


def generate_baseline(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    responses: list[str] = []
    batches = list(iter_batches(prompts, batch_size))

    for _, batch_prompts in tqdm(
        batches,
        desc="No-steering baseline",
    ):
        responses.extend(
            generate_batch(
                model=model,
                tokenizer=tokenizer,
                tokenized_inputs=tokenize_prompts(
                    tokenizer,
                    batch_prompts,
                ),
                max_new_tokens=max_new_tokens,
            )
        )

    return responses


def get_encoder_weight(sae: Any) -> torch.Tensor:
    weight = sae.W_enc.detach()

    n_features = (
        int(sae.cfg.d_sae)
        if hasattr(sae, "cfg") and hasattr(sae.cfg, "d_sae")
        else int(sae.W_dec.shape[0])
    )

    if weight.shape[0] == n_features:
        return weight

    if weight.shape[1] == n_features:
        return weight.T

    raise ValueError(
        f"Cannot infer W_enc orientation from shape {tuple(weight.shape)} "
        f"with {n_features} SAE features."
    )


def project_out_encoder_directions(
    target_direction: torch.Tensor,
    sae: Any,
    feature_ids: list[int],
    target_feature_idx: int | None,
    eps: float = 1e-8,
) -> torch.Tensor:
    original_ndim = target_direction.ndim
    vectors = (
        target_direction.unsqueeze(0)
        if original_ndim == 1
        else target_direction
    )

    if vectors.ndim != 2:
        raise ValueError(
            "target_direction must have shape [d_model] or "
            "[n_vectors, d_model]."
        )

    selected = sorted(
        {
            int(feature_idx)
            for feature_idx in feature_ids
            if feature_idx != target_feature_idx
        }
    )

    if not selected:
        return target_direction.clone()

    encoder = get_encoder_weight(sae).to(vectors.device, vectors.dtype)

    if min(selected) < 0 or max(selected) >= encoder.shape[0]:
        raise IndexError(
            "A collateral feature index is outside the SAE dictionary."
        )

    basis = encoder[selected]
    basis = basis[basis.norm(dim=1) > eps]

    if basis.shape[0] == 0:
        return target_direction.clone()

    q_matrix, _ = torch.linalg.qr(basis.T, mode="reduced")
    projected = vectors - (vectors @ q_matrix) @ q_matrix.T

    return (
        projected.squeeze(0)
        if original_ndim == 1
        else projected
    )


def direction_for_factor(
    run: SteeringRun,
    factor: float,
    collateral_ids_by_setting: dict[Any, Any] | None,
    analysis_name: str,
    topk: int,
) -> torch.Tensor:
    if collateral_ids_by_setting is None:
        return run.direction

    if run.sae is None:
        raise ValueError("Collateral projection requires an SAE.")

    collateral_ids = collateral_ids_by_setting.get(
        (analysis_name, factor),
        [],
    )
    selected_ids = [int(value) for value in collateral_ids[:topk]]

    print(f"factor={factor:g}: collateral feature IDs={selected_ids}")

    return project_out_encoder_directions(
        target_direction=run.direction,
        sae=run.sae,
        feature_ids=selected_ids,
        target_feature_idx=run.target_feature_idx,
    )


def run_factor_sweep(
    args: argparse.Namespace,
    model_base: Any,
    evaluation_set: EvaluationSet,
    baseline_responses: list[str],
    run: SteeringRun,
    collateral_ids_by_setting: dict[Any, Any] | None,
) -> list[dict[str, Any]]:
    tokenizer = model_base.tokenizer
    model = model_base.model
    rows: list[dict[str, Any]] = []

    for factor in tqdm(
        args.steering_factor,
        desc=f"{run.method} factor sweep",
    ):
        direction = direction_for_factor(
            run=run,
            factor=factor,
            collateral_ids_by_setting=collateral_ids_by_setting,
            analysis_name=evaluation_set.analysis_name,
            topk=args.topk,
        )

        batches = list(
            iter_batches(
                evaluation_set.prompts,
                args.batch_size,
            )
        )

        for start, batch_prompts in tqdm(
            batches,
            desc=f"factor={factor:g}",
            leave=False,
        ):
            steered_responses = generate_batch(
                model=model,
                tokenizer=tokenizer,
                tokenized_inputs=tokenize_prompts(
                    tokenizer,
                    batch_prompts,
                ),
                max_new_tokens=args.max_new_tokens,
                layer=run.layer,
                direction=direction,
                coefficient=factor,
                using_dim_vector=run.using_dim_vector,
            )

            for offset, (prompt, steered_response) in enumerate(
                zip(batch_prompts, steered_responses)
            ):
                dataset_index = start + offset

                rows.append(
                    {
                        "dataset": evaluation_set.name,
                        "subset": evaluation_set.split,
                        "dataset_index": dataset_index,
                        "prompt": prompt,
                        "ori_response": baseline_responses[dataset_index],
                        "steer_response": steered_response,
                        "factor": float(factor),
                        "method": run.method,
                        "sae_id": run.sae_id,
                        "feature_idx": run.feature_idx,
                        "rank_or_sample_id": run.sample_id,
                        "random_seed": run.random_seed,
                    }
                )

    return rows


def load_activation_feature_idx(model_path: str) -> int:
    path = ACTIVATION_STEERING_MAP.get(model_path)

    if path is None:
        raise ValueError(
            f"No activation feature configuration for {model_path}."
        )

    with path.open("r", encoding="utf-8") as handle:
        feature_info = json.load(handle)

    return int(
        feature_info["result"]["top_mean_activation_indices"][0]
    )


def load_ranked_feature_ids(
    model_path: str,
    topk: int = RANKED_FEATURE_TOPK,
) -> list[int]:
    """Load the first topk ranked SAE features for one model."""
    direction_path = DIM_VECTOR_MAP.get(model_path)

    if direction_path is None:
        raise ValueError(
            f"No DiM direction configuration for {model_path}."
        )

    ranked_path = direction_path.parent / "ranked_features.json"

    if not ranked_path.is_file():
        raise FileNotFoundError(
            f"ranked_features.json not found: {ranked_path}"
        )

    with ranked_path.open("r", encoding="utf-8") as handle:
        ranked_data = json.load(handle)

    records = ranked_data.get(model_path)

    if not isinstance(records, list):
        raise ValueError(
            f"Expected a feature list under key {model_path!r} "
            f"in {ranked_path}."
        )

    records = sorted(records, key=lambda record: int(record["rank"]))
    feature_ids = [
        int(record["feature_idx"])
        for record in records[:topk]
    ]

    if len(feature_ids) < topk:
        raise ValueError(
            f"{ranked_path} contains only {len(feature_ids)} features, "
            f"but requested topk={topk}."
        )

    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError(
            f"Duplicate ranked features found: {feature_ids}"
        )

    return feature_ids


def sample_random_feature_ids(
    n_features: int,
    num_samples: int,
    seed: int,
    excluded_ids: set[int],
    explicit_ids: list[int] | None,
) -> list[int]:
    candidates = [
        feature_idx
        for feature_idx in range(n_features)
        if feature_idx not in excluded_ids
    ]

    if explicit_ids:
        selected = [int(feature_idx) for feature_idx in explicit_ids]
        invalid = [
            feature_idx
            for feature_idx in selected
            if feature_idx not in candidates
        ]

        if invalid:
            raise ValueError(
                f"Invalid or excluded random feature IDs: "
                f"{sorted(set(invalid))}"
            )

        if len(set(selected)) != len(selected):
            raise ValueError(
                "--random_feature_ids must not contain duplicates."
            )

        return selected

    if num_samples > len(candidates):
        raise ValueError(
            f"Requested {num_samples} random features, but only "
            f"{len(candidates)} candidates are available."
        )

    return random.Random(seed).sample(candidates, k=num_samples)


def build_sae_runs(
    args: argparse.Namespace,
    sae: Any,
    sae_config: dict[str, Any],
) -> list[SteeringRun]:
    target_idx = int(sae_config["feature_idx"])
    layer = int(sae_config["target_layer"])
    sae_id = str(sae_config["sae_id"])
    n_features = int(sae.W_dec.shape[0])

    if args.method == "target":
        return [
            SteeringRun(
                method="target_sae",
                direction=sae.W_dec[target_idx].detach(),
                layer=layer,
                sae=sae,
                sae_id=sae_id,
                feature_idx=target_idx,
                target_feature_idx=target_idx,
            )
        ]

    if args.method == "activation":
        activation_idx = load_activation_feature_idx(args.model_path)

        return [
            SteeringRun(
                method="activation_sae",
                direction=sae.W_dec[activation_idx].detach(),
                layer=layer,
                sae=sae,
                sae_id=sae_id,
                feature_idx=activation_idx,
                target_feature_idx=target_idx,
            )
        ]

    if args.method == "ranked":
        feature_ids = load_ranked_feature_ids(
            model_path=args.model_path,
            topk=args.ranked_feature_topk,
        )

        invalid_ids = [
            feature_idx
            for feature_idx in feature_ids
            if feature_idx < 0 or feature_idx >= n_features
        ]

        if invalid_ids:
            raise IndexError(
                f"Ranked feature IDs outside SAE dictionary: {invalid_ids}. "
                f"SAE feature count: {n_features}."
            )

        print(
            f"Loaded top-{len(feature_ids)} ranked SAE features: "
            f"{feature_ids}"
        )

        return [
            SteeringRun(
                method="ranked_sae",
                direction=sae.W_dec[feature_idx].detach(),
                layer=layer,
                sae=sae,
                sae_id=sae_id,
                feature_idx=feature_idx,
                target_feature_idx=target_idx,
                sample_id=rank,
            )
            for rank, feature_idx in enumerate(feature_ids, start=1)
        ]

    if args.method == "random":
        activation_idx = load_activation_feature_idx(args.model_path)

        excluded_ids: set[int] = set()
        if not args.include_reference_features_in_random_pool:
            excluded_ids.update({target_idx, activation_idx})

        feature_ids = sample_random_feature_ids(
            n_features=n_features,
            num_samples=args.num_random_samples,
            seed=args.random_seed,
            excluded_ids=excluded_ids,
            explicit_ids=args.random_feature_ids,
        )

        print(
            f"Random SAE sampled {len(feature_ids)} feature(s) with seed "
            f"{args.random_seed}: {feature_ids}"
        )

        return [
            SteeringRun(
                method="random_sae",
                direction=sae.W_dec[feature_idx].detach(),
                layer=layer,
                sae=sae,
                sae_id=sae_id,
                feature_idx=feature_idx,
                target_feature_idx=target_idx,
                sample_id=sample_id,
                random_seed=args.random_seed,
            )
            for sample_id, feature_idx in enumerate(feature_ids)
        ]

    if args.method == "output_score":
        model_name = args.model_path.replace("/", "-")
        output_score_path = Path(f"{model_name}_output_scores.json")

        with output_score_path.open("r", encoding="utf-8") as handle:
            output_scores = json.load(handle)

        best_key, best_score = max(
            output_scores.items(),
            key=lambda item: item[1],
        )
        best_feature_idx = int(best_key.split("_")[1])

        print(f"Best output-score feature: {best_feature_idx}")
        print(f"Best output score: {best_score}")

        return [
            SteeringRun(
                method="output_score_sae",
                direction=sae.W_dec[best_feature_idx].detach(),
                layer=layer,
                sae=sae,
                sae_id=sae_id,
                feature_idx=best_feature_idx,
                target_feature_idx=target_idx,
            )
        ]

    raise ValueError(
        f"Unsupported SAE steering method: {args.method}"
    )


def load_dim_run(args: argparse.Namespace) -> SteeringRun:
    vector_path = DIM_VECTOR_MAP.get(args.model_path)
    metadata_path = DIM_VECTOR_METADATA_MAP.get(args.model_path)

    if vector_path is None or metadata_path is None:
        raise ValueError(
            f"No DiM vector configuration for {args.model_path}."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    direction = torch.load(vector_path, map_location="cpu")

    return SteeringRun(
        method="dim_vector",
        direction=direction,
        layer=int(metadata["layer"]) - 1,
        using_dim_vector=True,
    )


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def result_path(
    output_dir: Path,
    model_path: str,
    evaluation_set: EvaluationSet,
    run: SteeringRun,
) -> Path:
    parts = [
        sanitize_filename(model_path),
        run.method,
        f"use_safe_{evaluation_set.split == 'safe'}",
        f"steering_{evaluation_set.name}",
    ]

    if run.sae_id:
        parts.insert(1, sanitize_filename(run.sae_id))

    if run.method in {"random_sae", "ranked_sae"}:
        parts.extend(
            [
                f"rank_or_sample_{run.sample_id:03d}",
                f"feature_{run.feature_idx}",
            ]
        )

    return output_dir / ("_".join(parts) + ".json")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved: {path}")


def resolve_method(args: argparse.Namespace) -> str:
    legacy_methods = [
        ("random", args.use_random_feature),
        ("activation", args.use_activation_steering),
        ("dim", args.use_dim_vector),
    ]

    enabled = [
        method
        for method, is_enabled in legacy_methods
        if is_enabled
    ]

    if args.method is not None and enabled:
        raise ValueError(
            "Use either --method or a legacy method flag, not both."
        )

    if len(enabled) > 1:
        raise ValueError("Only one steering method can be enabled.")

    return args.method or (enabled[0] if enabled else "target")


def load_collateral_features(
    args: argparse.Namespace,
) -> dict[Any, Any] | None:
    if not args.use_collateral_projection:
        return None

    if args.method != "target":
        raise ValueError(
            "Collateral projection is supported only for Target SAE."
        )

    default_path = Path(
        f"{args.model_path.replace('/', '-')}_sae_feature_analysis_from_model/"
        f"{args.model_path.replace('/', '-')}_full_feature_stats_collateral_top20.pt"
    )

    path = args.collateral_path or default_path

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def get_sae_configs(
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not args.using_other_sae_config:
        try:
            return [SAE_CONFIG[args.model_path]]
        except KeyError as error:
            raise ValueError(
                f"No SAE configuration for {args.model_path}."
            ) from error

    with args.other_sae_config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        all_configs = json.load(handle)

    configs = all_configs.get(args.model_path)

    if not configs:
        raise ValueError(
            f"No alternative SAE configurations for {args.model_path}."
        )

    return configs[args.skip_other_sae_configs :]


def configure_tokenizer(tokenizer: Any) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"


def execute_runs(
    args: argparse.Namespace,
    model_base: Any,
    evaluation_set: EvaluationSet,
    baseline_responses: list[str],
    runs: list[SteeringRun],
    collateral_ids: dict[Any, Any] | None,
) -> list[dict[str, Any]]:
    manifest_entries: list[dict[str, Any]] = []

    for run in runs:
        rows = run_factor_sweep(
            args=args,
            model_base=model_base,
            evaluation_set=evaluation_set,
            baseline_responses=baseline_responses,
            run=run,
            collateral_ids_by_setting=collateral_ids,
        )

        output_path = result_path(
            output_dir=args.output_dir,
            model_path=args.model_path,
            evaluation_set=evaluation_set,
            run=run,
        )
        save_json(output_path, rows)

        manifest_entries.append(
            {
                "method": run.method,
                "model": args.model_path,
                "dataset": evaluation_set.name,
                "split": evaluation_set.split,
                "sae_id": run.sae_id,
                "rank_or_sample_id": run.sample_id,
                "feature_idx": run.feature_idx,
                "random_seed": run.random_seed,
                "output_file": str(output_path.resolve()),
            }
        )

    return manifest_entries


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run target, ranked, activation, random, output-score, "
            "or DiM steering sweeps."
        )
    )

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument(
        "--steering_dataset",
        choices=(
            "alpaca_eval",
            "wildguardmix",
            "xstest",
            "rewrite_alpaca",
        ),
        default="alpaca_eval",
    )

    parser.add_argument(
        "--steering_factor",
        type=float,
        nargs="+",
        default=DEFAULT_FACTORS,
    )

    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--use_safe", action="store_true")

    parser.add_argument(
        "--method",
        choices=(
            "target",
            "ranked",
            "random",
            "activation",
            "dim",
            "output_score",
        ),
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("steering_results"),
    )

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--ranked_feature_topk",
        type=int,
        default=RANKED_FEATURE_TOPK,
        help=(
            "Number of features loaded from ranked_features.json. "
            "Default: 5."
        ),
    )

    parser.add_argument("--num_random_samples", type=int, default=20)
    parser.add_argument(
        "--random_seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--random_feature_ids",
        type=int,
        nargs="+",
    )

    parser.add_argument(
        "--include_reference_features_in_random_pool",
        action="store_true",
        help="Allow Target SAE and Activation SAE IDs in random pool.",
    )

    parser.add_argument(
        "--use_collateral_projection",
        action="store_true",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--collateral_path", type=Path)

    parser.add_argument(
        "--using_other_sae_config",
        action="store_true",
    )
    parser.add_argument(
        "--other_sae_config_path",
        type=Path,
        default=Path("sae_config_release.json"),
    )
    parser.add_argument(
        "--skip_other_sae_configs",
        type=int,
        default=2,
        help="Skip this number of entries in other SAE configs.",
    )

    # Backward-compatible aliases.
    parser.add_argument("--use_random_feature", action="store_true")
    parser.add_argument(
        "--use_activation_steering",
        action="store_true",
    )
    parser.add_argument("--use_dim_vector", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = get_args()
    args.method = resolve_method(args)

    set_reproducibility(args.seed)

    cfg = Config(
        model_alias=os.path.basename(args.model_path),
        model_path=args.model_path,
    )

    model_base = construct_model_base(
        cfg.model_path,
        args.device,
    )
    model_base.model.eval()
    configure_tokenizer(model_base.tokenizer)

    evaluation_set = load_evaluation_set(
        dataset_name=args.steering_dataset,
        use_safe=args.use_safe,
    )

    baseline_responses = generate_baseline(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        prompts=evaluation_set.prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    collateral_ids = load_collateral_features(args)
    manifest_entries: list[dict[str, Any]] = []

    if args.method == "dim":
        manifest_entries.extend(
            execute_runs(
                args=args,
                model_base=model_base,
                evaluation_set=evaluation_set,
                baseline_responses=baseline_responses,
                runs=[load_dim_run(args)],
                collateral_ids=collateral_ids,
            )
        )
    else:
        for sae_config in get_sae_configs(args):
            print(f"Loading SAE: {sae_config['sae_id']}")

            sae = SAE.from_pretrained(
                sae_config["release"],
                sae_config["sae_id"],
                device=args.device,
            )
            sae.eval()

            runs = build_sae_runs(
                args=args,
                sae=sae,
                sae_config=sae_config,
            )

            manifest_entries.extend(
                execute_runs(
                    args=args,
                    model_base=model_base,
                    evaluation_set=evaluation_set,
                    baseline_responses=baseline_responses,
                    runs=runs,
                    collateral_ids=collateral_ids,
                )
            )

            del sae

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    manifest_path = (
        args.output_dir
        / (
            f"{sanitize_filename(args.model_path)}_"
            f"{args.method}_"
            f"use_safe_{args.use_safe}_"
            f"steering_{args.steering_dataset}_manifest.json"
        )
    )
    save_json(manifest_path, manifest_entries)


if __name__ == "__main__":
    main()
