import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sae_lens import SAE

from pipeline.utils.sae_config import SAE_CONFIG


MODEL_LIST = [
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]

DIM_VECTOR_MAP = {
    "google/gemma-3-1b-it": Path(
        "pipeline/runs/gemma-3-1b-it/direction.pt"
    ),
    "google/gemma-3-4b-it": Path(
        "pipeline/runs/gemma-3-4b-it/direction.pt"
    ),
    "google/gemma-3-12b-it": Path(
        "pipeline/runs/gemma-3-12b-it/direction.pt"
    ),
    "google/gemma-3-27b-it": Path(
        "pipeline/runs/gemma-3-27b-it/direction.pt"
    ),
}

DEVICE = "cuda:3"
TOPK = 10


def main() -> None:
    for model_name in MODEL_LIST:
        if model_name not in SAE_CONFIG:
            raise ValueError(f"No SAE configuration for {model_name}")

        dim_vec_path = DIM_VECTOR_MAP[model_name]
        if not dim_vec_path.is_file():
            raise FileNotFoundError(
                f"DiM direction vector not found: {dim_vec_path}"
            )

        sae_config = SAE_CONFIG[model_name]

        print(f"\nLoading SAE for {model_name}: {sae_config['sae_id']}")

        sae = SAE.from_pretrained(
            sae_config["release"],
            sae_config["sae_id"],
            device=DEVICE,
        )
        sae.eval()

        dim_vec = torch.load(
            dim_vec_path,
            map_location=DEVICE,
        ).squeeze()

        dim_vec = dim_vec.to(
            device=sae.W_dec.device,
            dtype=sae.W_dec.dtype,
        )

        if dim_vec.ndim != 1:
            raise ValueError(
                f"{model_name}: expected a 1D DiM vector, "
                f"got shape {tuple(dim_vec.shape)}"
            )

        if dim_vec.shape[0] != sae.W_dec.shape[1]:
            raise ValueError(
                f"{model_name}: DiM vector shape {tuple(dim_vec.shape)} "
                f"does not match SAE decoder shape "
                f"{tuple(sae.W_dec.shape)}"
            )

        with torch.no_grad():
            similarities = F.cosine_similarity(
                sae.W_dec,
                dim_vec.unsqueeze(0),
                dim=-1,
            )

            top_values, top_indices = torch.topk(
                similarities,
                k=TOPK,
                largest=True,
            )

        ranked_features = [
            {
                "rank": rank,
                "feature_idx": int(feature_idx),
                "cos_sim": float(cos_sim),
            }
            for rank, (feature_idx, cos_sim) in enumerate(
                zip(top_indices.tolist(), top_values.tolist()),
                start=1,
            )
        ]

        print(f"Top {TOPK} features for {model_name}:")
        for record in ranked_features:
            print(
                f"rank={record['rank']}, "
                f"feature_idx={record['feature_idx']}, "
                f"cos_sim={record['cos_sim']:.6f}"
            )

        # 与 direction.pt 保存在同一个模型目录。
        output_path = dim_vec_path.parent / "ranked_features.json"

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {model_name: ranked_features},
                handle,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Saved rankings to: {output_path}")

        del sae, dim_vec, similarities, top_values, top_indices

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
