#!/usr/bin/env python3

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from sae_lens import SAE
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

SAE_CONFIG = {
    "google/gemma-3-1b-it": {
        "release": "gemma-scope-2-1b-it-res-all",
        "sae_id": "layer_16_width_16k_l0_big",
        "target_layer": 16,
    },
    "google/gemma-3-4b-it": {
        "release": "gemma-scope-2-4b-it-res-all",
        "sae_id": "layer_16_width_16k_l0_big",
        "target_layer": 16,
    },
    "google/gemma-3-12b-it": {
        "release": "gemma-scope-2-12b-it-res-all",
        "sae_id": "layer_28_width_262k_l0_big",
        "target_layer": 28,
    },
    "google/gemma-3-27b-it": {
        "release": "gemma-scope-2-27b-it-res-all",
        "sae_id": "layer_39_width_262k_l0_big",
        "target_layer": 39,
    },
}


FEATURE_FILE_PATH = {
    "google/gemma-3-1b-it": (
        "gemma-3-1b-it_topkactivation_layer16.json"
    ),
    "google/gemma-3-4b-it": (
        "gemma-3-4b-it_topkactivation_layer16.json"
    ),
    "google/gemma-3-12b-it": (
        "gemma-3-12b-it_topkactivation_layer28.json"
    ),
    "google/gemma-3-27b-it": (
        "gemma-3-27b-it_topkactivation_layer39.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=tuple(SAE_CONFIG.keys()),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:5",
    )

    parser.add_argument(
        "--logit_lens_top_k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--amp_factor",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--neutral_sentence",
        type=str,
        default="From my experience,",
    )

    parser.add_argument(
        "--features_file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--cache_path",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def get_text_backbone(model: Any) -> Any:
    """兼容 Gemma-3-1B 和 Gemma-3-4B/12B/27B 的模型结构。"""

    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
    ):
        return model.model.language_model

    if hasattr(model, "language_model"):
        return model.language_model

    if hasattr(model, "model"):
        return model.model

    raise ValueError(
        f"Cannot locate text backbone for {type(model).__name__}"
    )


def get_lm_head(
    model: Any,
    text_backbone: Any,
) -> torch.nn.Module:
    if hasattr(model, "lm_head"):
        return model.lm_head

    if hasattr(text_backbone, "lm_head"):
        return text_backbone.lm_head

    raise ValueError("Cannot locate lm_head.")


def load_sae(
    model_name: str,
    device: str,
) -> SAE:
    config = SAE_CONFIG[model_name]

    loaded = SAE.from_pretrained(
        release=config["release"],
        sae_id=config["sae_id"],
        device=device,
    )

    # 兼容不同 sae-lens 版本
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    sae.eval()

    return sae


def get_features_by_layers(
    features_file: Path,
) -> dict[int, list[int]]:
    with features_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    layer = int(data["target_layer"])

    feature_ids = [
        int(feature)
        for feature in data["result"][
            "top_mean_activation_indices"
        ]
    ]

    if not feature_ids:
        raise ValueError(
            f"No features found in {features_file}"
        )

    return {layer: feature_ids}


class AmplifySAEHook:
    """在最后一个 token 位置放大一个 SAE feature。"""

    def __init__(
        self,
        sae: SAE,
        feature_idx: int,
        amp_factor: float,
    ) -> None:
        self.sae = sae
        self.feature_idx = feature_idx
        self.amp_factor = amp_factor

    @torch.no_grad()
    def __call__(
        self,
        module: Any,
        inputs: Any,
        output: Any,
    ) -> torch.Tensor:
        del module, inputs

        # Gemma-3 通常直接返回 Tensor。
        # 兼容少数 Transformers 版本返回 tuple 的情况。
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(
                "Expected Tensor layer output, "
                f"got {type(hidden_states).__name__}"
            )

        if hidden_states.ndim != 3:
            raise ValueError(
                "Expected [batch, seq_len, hidden_dim], "
                f"got {tuple(hidden_states.shape)}"
            )

        sae_device = self.sae.W_enc.device
        sae_dtype = self.sae.W_enc.dtype

        # 模型通常为 bfloat16，SAE 通常为 float32。
        hidden_for_sae = hidden_states.to(
            device=sae_device,
            dtype=sae_dtype,
        )

        feature_acts = self.sae.encode(hidden_for_sae)

        clean_reconstruction = self.sae.decode(
            feature_acts
        )

        sae_error = (
            hidden_for_sae - clean_reconstruction
        )

        feature_acts = feature_acts.clone()

        # 和原始 Arad output-score 实现保持一致。
        max_act_value = feature_acts[
            :, -1, :
        ].max().item()

        feature_acts[
            :, -1, self.feature_idx
        ] += max_act_value * self.amp_factor

        steered_hidden = (
            self.sae.decode(feature_acts)
            + sae_error
        )

        # Gemma-3 下一层必须接收 Tensor。
        return steered_hidden.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )


@torch.inference_mode()
def cache_candidate_logit_lens(
    sae: SAE,
    feature_ids: list[int],
    final_layer_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    k: int,
) -> dict[int, list[int]]:
    """
    只对候选 features 计算 Logit Lens，
    避免对全部 SAE features 构建巨大 logits 矩阵。
    """

    max_feature_idx = sae.W_dec.shape[0] - 1

    invalid_features = [
        feature
        for feature in feature_ids
        if feature < 0 or feature > max_feature_idx
    ]

    if invalid_features:
        raise ValueError(
            f"Invalid feature IDs: {invalid_features}"
        )

    target_device = lm_head.weight.device
    target_dtype = lm_head.weight.dtype

    decoder_weights = sae.W_dec[
        feature_ids
    ].detach().to(
        device=target_device,
        dtype=target_dtype,
    )

    decoder_weights = final_layer_norm(
        decoder_weights
    ).to(
        device=target_device,
        dtype=target_dtype,
    )

    logits = lm_head(decoder_weights)

    # softmax 不影响 token 排序，直接对 logits 求 top-k。
    topk_token_ids = torch.topk(
        logits,
        k=k,
        dim=-1,
    ).indices.cpu()

    return {
        feature_idx: topk_token_ids[position].tolist()
        for position, feature_idx in enumerate(feature_ids)
    }


@torch.inference_mode()
def get_output_score(
    layer: int,
    feature_idx: int,
    logit_lens_token_ids: list[int],
    sentence: str,
    sae: SAE,
    tokenizer: Any,
    model: Any,
    text_backbone: Any,
    device: str,
    amp_factor: float,
) -> float:
    """
    返回单个 float：

    output_score =
        (1 - best_rank / vocab_size)
        * probability(best_logit_lens_token)
    """

    target_layer = text_backbone.layers[layer]

    sae_hook = AmplifySAEHook(
        sae=sae,
        feature_idx=feature_idx,
        amp_factor=amp_factor,
    )

    handle = target_layer.register_forward_hook(
        sae_hook,
        always_call=True,
    )

    try:
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        outputs = model(
            **inputs,
            use_cache=False,
        )

    finally:
        handle.remove()

    # 保留 bfloat16 softmax 行为，与原始代码更一致。
    intervention_logits = outputs.logits[0, -1]

    intervention_probs = torch.softmax(
        intervention_logits,
        dim=0,
    )

    token_ids = torch.tensor(
        logit_lens_token_ids,
        device=intervention_logits.device,
        dtype=torch.long,
    )

    candidate_logits = intervention_logits[token_ids]

    best_position = torch.argmax(candidate_logits)
    best_token_id = token_ids[best_position]

    best_logit = intervention_logits[best_token_id]

    # 不使用 argsort，直接计算 rank，节省显存和时间。
    best_rank = (
        intervention_logits > best_logit
    ).sum().item()

    vocab_size = intervention_probs.shape[0]

    top_token_score = intervention_probs[
        best_token_id
    ].item()

    rank_output_score = (
        1.0 - best_rank / vocab_size
    )

    return float(
        rank_output_score * top_token_score
    )


def save_json(
    path: Path,
    output_scores: dict[str, float],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_scores,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(path)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = args.device
    model_name = args.model_name

    if args.features_file is None:
        features_file = Path(
            FEATURE_FILE_PATH[model_name]
        )
    else:
        features_file = args.features_file

    if args.cache_path is None:
        cache_path = Path(
            f"./{model_name.replace('/', '_')}_"
            "output_scores.json"
        )
    else:
        cache_path = args.cache_path

    if not features_file.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_file}"
        )

    print(args)
    print(f"Loading model: {model_name}")

    features_by_layers = get_features_by_layers(
        features_file
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()

    text_backbone = get_text_backbone(model)
    final_layer_norm = text_backbone.norm
    lm_head = get_lm_head(
        model,
        text_backbone,
    )

    print("Model:", type(model).__name__)
    print("Backbone:", type(text_backbone).__name__)
    print("Layers:", len(text_backbone.layers))
    print("Final norm:", final_layer_norm.weight.shape)
    print("LM head:", lm_head.weight.shape)

    sae = load_sae(
        model_name=model_name,
        device=device,
    )

    print("SAE decoder:", sae.W_dec.shape)

    if cache_path.exists():
        with cache_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            output_scores = json.load(file)

        # 确保旧缓存也是简单 float 格式。
        output_scores = {
            key: float(value)
            for key, value in output_scores.items()
        }
    else:
        output_scores = {}

    expected_layer = SAE_CONFIG[
        model_name
    ]["target_layer"]

    for layer, features in features_by_layers.items():
        if layer != expected_layer:
            raise ValueError(
                f"Feature-file layer={layer}, "
                f"SAE_CONFIG layer={expected_layer}"
            )

        if layer >= len(text_backbone.layers):
            raise ValueError(
                f"Invalid layer={layer}; "
                f"model has {len(text_backbone.layers)} layers."
            )

        # 去重，同时保持原顺序。
        features = list(dict.fromkeys(
            int(feature)
            for feature in features
        ))

        print(
            f"Layer {layer}: "
            f"{len(features)} candidate features."
        )

        feature_to_logit_lens_ids = (
            cache_candidate_logit_lens(
                sae=sae,
                feature_ids=features,
                final_layer_norm=final_layer_norm,
                lm_head=lm_head,
                k=args.logit_lens_top_k,
            )
        )

        for feature_idx in tqdm(
            features,
            desc=f"Layer {layer} output scores",
        ):
            layer_feature_key = (
                f"{layer}_{feature_idx}"
            )

            # 支持中断后继续运行。
            if layer_feature_key in output_scores:
                continue

            output_score = get_output_score(
                layer=layer,
                feature_idx=feature_idx,
                logit_lens_token_ids=(
                    feature_to_logit_lens_ids[
                        feature_idx
                    ]
                ),
                sentence=args.neutral_sentence,
                sae=sae,
                tokenizer=tokenizer,
                model=model,
                text_backbone=text_backbone,
                device=device,
                amp_factor=args.amp_factor,
            )

            # 输出 JSON 只保存 key -> float。
            output_scores[
                layer_feature_key
            ] = output_score

            save_json(
                cache_path,
                output_scores,
            )

    ranked_scores = sorted(
        output_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    print("\nTop output scores:")

    for layer_feature_key, output_score in ranked_scores[:20]:
        print(
            f"{layer_feature_key}: "
            f"{output_score:.8f}"
        )

    del sae
    del model

    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nSaved output scores to: {cache_path}")


if __name__ == "__main__":
    main()
