#!/usr/bin/env python3
"""Export merged Qwen3.8 BF16 weights in Qwen's native fine-grained FP8 layout.

This exporter intentionally does not instantiate the Transformers model.  It
processes one safetensors shard and one weight tensor at a time, which makes it
possible to create the native ``quant_method=fp8`` checkpoint on an Ada L40S.
The block quantization formula and ``weight_scale_inv`` convention match
Transformers' ``Fp8Quantize`` implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


BLOCK_SIZE = (128, 128)
EXPECTED_MODEL_TYPE = "qwen3_5"
EXPECTED_ARCHITECTURES = ["Qwen3_5ForConditionalGeneration"]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_configs(source: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    for key, expected in (
        ("model_type", EXPECTED_MODEL_TYPE),
        ("architectures", EXPECTED_ARCHITECTURES),
    ):
        if source.get(key) != expected:
            raise ValueError(f"Merged source has {key}={source.get(key)!r}; expected {expected!r}")
        if reference.get(key) != expected:
            raise ValueError(f"FP8 reference has {key}={reference.get(key)!r}; expected {expected!r}")
    if source.get("quantization_config") is not None:
        raise ValueError("Merged source is already quantized")

    for section, keys in (
        ("text_config", ("num_hidden_layers", "hidden_size", "vocab_size", "layer_types")),
        ("vision_config", ("depth", "hidden_size", "out_hidden_size")),
    ):
        source_section = source.get(section, {})
        reference_section = reference.get(section, {})
        for key in keys:
            if source_section.get(key) != reference_section.get(key):
                raise ValueError(f"Source/reference {section} mismatch for {key}")

    quant = reference.get("quantization_config")
    if not isinstance(quant, dict) or quant.get("quant_method") != "fp8":
        raise ValueError("Reference must use native quant_method=fp8")
    if quant.get("activation_scheme") != "dynamic":
        raise ValueError("Reference must use dynamic activation quantization")
    if quant.get("weight_block_size") != list(BLOCK_SIZE):
        raise ValueError(f"Reference must use weight_block_size={list(BLOCK_SIZE)}")
    ignored = quant.get("modules_to_not_convert")
    if not isinstance(ignored, list) or not ignored or not all(
        isinstance(item, str) and item for item in ignored
    ):
        raise ValueError("Reference has no valid modules_to_not_convert list")
    for family in ("lm_head", "embed_tokens", "linear_attn", "model.visual", "mtp"):
        if not any(family in item for item in ignored):
            raise ValueError(f"Reference does not preserve required family: {family}")
    return list(ignored)


def tensor_action(
    name: str,
    ndim: int,
    shape: tuple[int, ...],
    official_quantized_weights: set[str],
) -> str:
    """Follow the official checkpoint's scale-tensor set exactly."""
    if name not in official_quantized_weights:
        return "copy"
    if not name.endswith(".weight") or ndim != 2:
        raise ValueError(f"Official FP8 oracle selected a non-2-D weight: {name} shape={shape}")
    if shape[-2] % BLOCK_SIZE[0] or shape[-1] % BLOCK_SIZE[1]:
        raise ValueError(
            f"Unignored 2-D weight is not divisible by {BLOCK_SIZE}: {name} shape={shape}"
        )
    return "quantize"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-model-dir",
        type=Path,
        required=True,
        help="Complete official Qwen FP8 checkpoint; its index is the quantization oracle",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-relative-l2-error",
        type=float,
        default=0.05,
        help="Per-tensor ||W-dequant(Q(W))||_2 / ||W||_2 acceptance ceiling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    reference_model_dir = args.reference_model_dir.resolve()
    reference_config_path = reference_model_dir / "config.json"
    reference_index_path = reference_model_dir / "model.safetensors.index.json"
    output_dir = args.output_dir.resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(f"Refusing to reuse output or staging path: {output_dir}")
    if not 0 < args.max_relative_l2_error < 1:
        raise ValueError("--max-relative-l2-error must be between zero and one")

    source_config_path = model_dir / "config.json"
    source_index_path = model_dir / "model.safetensors.index.json"
    for path in (
        source_config_path,
        source_index_path,
        reference_config_path,
        reference_index_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_config = load_json(source_config_path)
    reference_config = load_json(reference_config_path)
    ignored = validate_configs(source_config, reference_config)
    source_index = load_json(source_index_path)
    weight_map = source_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid safetensors weight map: {source_index_path}")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in weight_map.items()):
        raise ValueError("Safetensors weight_map must map strings to strings")
    if any(name.endswith(".weight_scale_inv") for name in weight_map):
        raise ValueError("Source unexpectedly already contains native FP8 scale tensors")
    reference_index = load_json(reference_index_path)
    reference_weight_map = reference_index.get("weight_map")
    if not isinstance(reference_weight_map, dict) or not reference_weight_map:
        raise ValueError(f"Invalid official FP8 weight map: {reference_index_path}")
    scale_suffix = ".weight_scale_inv"
    weight_suffix = ".weight"
    reference_scales = {
        str(name) for name in reference_weight_map if str(name).endswith(scale_suffix)
    }
    official_quantized_weights = {
        name[: -len(scale_suffix)] + weight_suffix for name in reference_scales
    }
    if not official_quantized_weights:
        raise ValueError("Official FP8 checkpoint has no weight_scale_inv tensors")
    reference_base_tensors = set(reference_weight_map) - reference_scales
    source_tensors = set(weight_map)
    missing_reference_tensors = reference_base_tensors - source_tensors
    extra_source_tensors = source_tensors - reference_base_tensors
    disallowed_missing = {
        name for name in missing_reference_tensors if not name.startswith("mtp.")
    }
    if extra_source_tensors or disallowed_missing:
        raise ValueError(
            "Merged BF16 tensor set does not match official Qwen FP8 architecture: "
            f"non_mtp_missing={sorted(disallowed_missing)[:5]} "
            f"extra={sorted(extra_source_tensors)[:5]}"
        )
    # MS-SWIFT's merged inference checkpoint omits the optional MTP draft head.
    # It is unused unless speculative decoding is explicitly enabled. Apply the
    # official quantization selection to every weight that the merged target
    # actually contains, while allowing only those absent ``mtp.*`` tensors.
    selected_quantized_weights = official_quantized_weights & source_tensors
    if not selected_quantized_weights:
        raise ValueError("No official quantized weights are present in the BF16 source")

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("This exporter requires torch and safetensors") from error
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("Installed PyTorch has no float8_e4m3fn dtype")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested but torch.cuda.is_available() is false")

    shards: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        shards.setdefault(shard_name, []).append(tensor_name)
    staging_dir.mkdir(parents=True)

    output_weight_map: dict[str, str] = {}
    tensor_stats: list[dict[str, Any]] = []
    total_size = 0
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = float(torch.finfo(fp8_dtype).max)
    fp8_min = float(torch.finfo(fp8_dtype).min)

    for shard_number, shard_name in enumerate(sorted(shards), 1):
        source_shard = model_dir / shard_name
        if not source_shard.is_file():
            raise FileNotFoundError(source_shard)
        output_tensors: dict[str, Any] = {}
        print(f"[{shard_number}/{len(shards)}] {shard_name}", flush=True)
        with safe_open(source_shard, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            expected = set(shards[shard_name])
            if available != expected:
                raise ValueError(
                    f"Index/shard key mismatch for {shard_name}: "
                    f"missing={sorted(expected - available)[:3]} extra={sorted(available - expected)[:3]}"
                )
            for name in sorted(expected):
                source_tensor = handle.get_tensor(name)
                shape = tuple(source_tensor.shape)
                action = tensor_action(
                    name, source_tensor.ndim, shape, selected_quantized_weights
                )
                if action == "copy":
                    output_tensors[name] = source_tensor.contiguous()
                    output_weight_map[name] = shard_name
                    continue

                value = source_tensor.to(device=device, dtype=torch.float32)
                rows, cols = shape
                reshaped = value.reshape(
                    rows // BLOCK_SIZE[0], BLOCK_SIZE[0],
                    cols // BLOCK_SIZE[1], BLOCK_SIZE[1],
                )
                max_abs = reshaped.abs().amax(dim=(1, 3))
                safe_max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
                forward_scale = fp8_max / safe_max_abs
                forward_scale = torch.where(max_abs > 0, forward_scale, torch.ones_like(forward_scale))
                scale_inv = (1.0 / forward_scale).to(torch.float32)
                scaled = reshaped * forward_scale.unsqueeze(1).unsqueeze(3)
                quantized = torch.clamp(scaled, min=fp8_min, max=fp8_max).to(fp8_dtype)
                quantized = quantized.reshape(shape)

                dequantized = quantized.to(torch.float32).reshape_as(reshaped)
                dequantized.mul_(scale_inv.unsqueeze(1).unsqueeze(3))
                difference = dequantized - reshaped
                denominator = torch.linalg.vector_norm(reshaped)
                numerator = torch.linalg.vector_norm(difference)
                relative_l2_error = float((numerator / denominator.clamp_min(1e-30)).item())
                if not torch.isfinite(scale_inv).all() or (scale_inv <= 0).any():
                    raise RuntimeError(f"Invalid FP8 scales for {name}")
                if relative_l2_error > args.max_relative_l2_error:
                    raise RuntimeError(
                        f"FP8 reconstruction error too high for {name}: {relative_l2_error:.6g}"
                    )

                scale_name = name[: -len(".weight")] + ".weight_scale_inv"
                output_tensors[name] = quantized.cpu().contiguous()
                output_tensors[scale_name] = scale_inv.cpu().contiguous()
                output_weight_map[name] = shard_name
                output_weight_map[scale_name] = shard_name
                tensor_stats.append(
                    {"name": name, "shape": list(shape), "relative_l2_error": relative_l2_error}
                )
                del value, reshaped, max_abs, safe_max_abs, forward_scale
                del scaled, quantized, dequantized, difference, denominator, numerator, scale_inv
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        output_shard = staging_dir / shard_name
        save_file(output_tensors, output_shard, metadata={"format": "pt"})
        total_size += sum(tensor.numel() * tensor.element_size() for tensor in output_tensors.values())
        del output_tensors

    if not tensor_stats:
        raise RuntimeError("No tensors were quantized")
    quantized_names = {item["name"] for item in tensor_stats}
    if quantized_names != selected_quantized_weights:
        raise RuntimeError(
            "Export did not reproduce official quantized tensor selection: "
            f"missing={sorted(selected_quantized_weights - quantized_names)[:5]} "
            f"extra={sorted(quantized_names - selected_quantized_weights)[:5]}"
        )

    for source_file in model_dir.iterdir():
        if not source_file.is_file():
            continue
        if source_file.name == "config.json" or source_file.name == "model.safetensors.index.json":
            continue
        if source_file.suffix == ".safetensors":
            continue
        shutil.copy2(source_file, staging_dir / source_file.name)

    output_config = dict(source_config)
    output_config["quantization_config"] = dict(reference_config["quantization_config"])
    (staging_dir / "config.json").write_text(
        json.dumps(output_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    (staging_dir / "model.safetensors.index.json").write_text(
        json.dumps(output_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": "orena-qwen38-native-fp8-export-v1",
        "source_model": str(model_dir),
        "reference_config": str(reference_config_path),
        "reference_config_sha256": sha256(reference_config_path),
        "reference_index": str(reference_index_path),
        "reference_index_sha256": sha256(reference_index_path),
        "official_quantized_weight_count": len(official_quantized_weights),
        "selected_quantized_weight_count": len(selected_quantized_weights),
        "missing_optional_mtp_tensors": sorted(missing_reference_tensors),
        "quantized_tensors": len(tensor_stats),
        "max_relative_l2_error": max(item["relative_l2_error"] for item in tensor_stats),
        "mean_relative_l2_error": (
            sum(item["relative_l2_error"] for item in tensor_stats) / len(tensor_stats)
        ),
        "weight_bytes": total_size,
        "tensor_stats": tensor_stats,
    }
    (staging_dir / "orena_native_fp8_export_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging_dir.rename(output_dir)
    print(json.dumps({key: value for key, value in metadata.items() if key != "tensor_stats"}, indent=2))
    print(f"Completed native FP8 export: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
