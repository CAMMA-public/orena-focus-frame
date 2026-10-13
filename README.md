# CAMMA Qwen3.8 FRAME

Code for CAMMA's single-frame surgical visual-question-answering submission to
the ORena FOCUS FRAME track. The final model adapts **Qwen3.8-27B** with
rank-32 LoRA on all linear layers in the vision encoder and language decoder,
while training the vision-language merger in full. The repository contains the training recipe, LoRA merge utility, native FP8
exporter, and the final offline Docker inference implementation. 

## Important points about our method

- One image per question without temporal overlay.
- Qwen3.8-27B initialized from the unquantized BF16 checkpoint.
- Vision encoder: LoRA, rank 32, alpha 64, learning rate `1e-5`.
- Language decoder: LoRA, rank 32, alpha 64, learning rate `5e-5`.
- Vision-language merger: fully trained at `1e-5`.
- One epoch, effective batch size 8, BF16 training, sequence length 2,048.
- Concise non-thinking prompt and deterministic decoding.
- LoRA adapters merged into the base model after training.
- Deployment checkpoint exported to Qwen's native mixed FP8/BF16 layout and
  served with vLLM 0.21.0 and InstantTensor.

## Training data

The final run used 48,545 image-question-answer records. We included all
available labelled FOCUS FRAME examples and capped auxiliary subsets to avoid
one dataset dominating training.

| Dataset | Rows used | Public source |
| --- | ---: | --- |
| ORena FOCUS FRAME (HeiCo and LapChole) | 20,000 | [Challenge](https://frame.orena-focus-challenge.org/) · [loader utilities](https://github.com/IMSY-DKFZ/orena-focus) · [HeiCo annotations](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa) · [LapChole annotations](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa) |
| Cholec80-VQA | 8,000 | [Surgical-VQA](https://github.com/lalithjets/Surgical_VQA) |
| SSG-VQA | 6,000 | [SSG-VQA](https://github.com/CAMMA-public/SSG-VQA) |
| Surgical-VQLA++ | 6,000 | [Surgical-VQLA++](https://github.com/longbai1006/Surgical-VQLAPlus) |
| PitVQA | 4,000 | [PitVQA](https://github.com/mobarakol/PitVQA) |
| Kvasir-VQA | 4,000 | [Kvasir-VQA](https://github.com/simula/Kvasir-VQA) · [dataset page](https://datasets.simula.no/kvasir-vqa/) |
| DAISI-VQA | 545 | [LLM-CL-VQA](https://github.com/yuyangdu01/llm-cl-vqa) |

Download each dataset from its original publisher and follow its access terms,
license, and citation requirements. FOCUS challenge data may be subject to its
own access agreement and must not be redistributed. This repository therefore
does not include private directory layouts or machine-specific extraction
scripts. Each selected annotation must be resolved to its corresponding image
locally before training.

The final selection included all 20,000 FOCUS examples and deterministic,
group-aware capped samples from the auxiliary sources. When reproducing the
dataset, avoid duplicate image-question pairs, keep all questions associated
with their correct source image, and record the sampling seed (`20260728`) and
selected IDs. Because the final model trained on all labelled public FOCUS
data, evaluation on those same examples is training-contaminated and must not
be reported as clean validation.

## Expected training file

Prepare one newline-delimited JSON file in the native MS-SWIFT multimodal
format. Image paths may be absolute or otherwise resolvable from the training
working directory. Each line should resemble:

```json
{"images":["/path/to/frame.jpg"],"messages":[{"role":"system","content":"You are a surgical assistant. You are given endoscopic video from a minimally invasive procedure. Analyze the footage and answer the surgical question based on the visual evidence. Be precise and concise.\n\n<foreign-object definitions when applicable>"},{"role":"user","content":"<image>\nProcedure: <procedure type>\nQuestion: <question>"},{"role":"assistant","content":"<reference answer>"}]}
```

Validate that there is exactly one readable image and one non-empty answer per
record before starting an expensive run. The supplied code intentionally starts
from this prepared JSONL rather than downloading or transforming datasets.

## Environment

Training was performed with:

- Python 3.12
- PyTorch 2.13.0 with CUDA support
- MS-SWIFT 4.4.2
- Transformers 5.12.1
- PEFT 0.19.1
- qwen-vl-utils 0.0.14

Install a PyTorch build appropriate for your CUDA driver first, then install
the packages in [`training/requirements.txt`](training/requirements.txt). The
exact final single-GPU recipe requires a BF16-capable GPU with approximately
80 GB of memory. Multi-GPU adaptation requires an appropriate FSDP/DeepSpeed
configuration and is outside this minimal reproduction.

Download the unquantized
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) checkpoint and
accept any model access conditions shown by Hugging Face. Do not use the FP8
checkpoint as the training base.

## Train and merge

Run the one-epoch final recipe:

```bash
bash training/train_qwen38_all_public.sh \
  /path/to/Qwen3.8-27B \
  /path/to/all_public_submission_train.jsonl \
  /path/to/training-output
```

The output directory may contain versioned subdirectories. Select the final
adapter checkpoint containing `adapter_config.json`, then merge it:

```bash
bash training/merge_qwen38_adapter.sh \
  /path/to/Qwen3.8-27B \
  /path/to/final-adapter-checkpoint \
  /path/to/merged-bf16-model
```

Both scripts reject missing inputs and existing output paths rather than
silently overwriting them.

## Native FP8 export

Deployment used Qwen's official FP8 checkpoint as an architecture and tensor
layout reference. Download
[`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8), then run:

```bash
bash training/export_qwen38_native_fp8.sh \
  /path/to/merged-bf16-model \
  /path/to/Qwen3.8-27B-FP8 \
  /path/to/fine-tuned-native-fp8-model
```

The exporter processes one tensor/shard at a time and validates the output. It
requires a CUDA-enabled PyTorch build with `float8_e4m3fn` support and the
`safetensors` package. The official FP8 weights are used only as a layout
oracle; the exported values come from the merged fine-tuned model.

## Build the FRAME container

The `submission/` directory mirrors the final challenge runtime. Model weights
are deliberately ignored by Git. Copy the exported checkpoint into the empty
model directory:

```bash
rsync -a /path/to/fine-tuned-native-fp8-model/ submission/resources/model/
cd submission
bash do_build.sh
```

For a local GPU smoke test, prepare the official FRAME container input layout
under `submission/test/input/interface_1/` (`request.json`,
`FO_definitions.json`, and `frames/<qID>.png`), then run:

```bash
bash do_test_run.sh
```

The container is Linux/AMD64, fully offline at inference, and expects the
NVIDIA Container Toolkit. Its vLLM cold-start adjustments were developed for
the challenge's L40S deployment environment and are tightly coupled to vLLM
0.21.0; retest latency and correctness before changing the serving stack.

## Weights

The fine-tuned mixed FP8/BF16 checkpoint is available on Hugging Face:

[CAMMA Qwen3.8-27B FOCUS FRAME FP8](https://huggingface.co/lalithnag/CAMMA-Qwen3.8-27B-FOCUS-FRAME-FP8)

The model is licensed in the same terms as the Qwen3.8 model.

## Reproducibility notes

- Random/data seed: `20260728`.
- Exactly one image is supplied for each question.
- Training and inference disable thinking output.
- Inference uses greedy decoding (`temperature=0`, `top_p=1`, `top_k=-1`) and
  generates at most 128 tokens.
- Images are capped at `768 × 768` pixels by the multimodal processor.
- This is research software for benchmark use, not a clinical system.

## Citation

Please cite the original datasets and Qwen in accordance with their respective
repositories. The project-specific citation will be provided with the public
FOCUS challenge report.
