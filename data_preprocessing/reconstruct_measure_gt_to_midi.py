import argparse
from typing import List, Sequence, Tuple

import numpy as np
import safetensors.torch
import torch
from transformers import LlamaConfig

from config import ModelConfig
from model import PianoLLaMA
from PianoDataset import encode_measure_tokens
from Token2Midi import tokens_to_midi
from inference import save_gt_midi
from my_tokenizer import PianoRollTokenizer
#reconstruct_measure_gt_to_midi.py --model-path="/home/cby/not_use/Advanced/generative_newtoken_improved_1_4_relative_track_RT_Compress_measure/checkpoints/steps_30000_1107_0001/model.safetensors" --gt-path="/DATA2_4T/cby/home/lab-wei.zhenao/boyu/Dataset/allxml_npz_dual_track_optimized/1259016.npz"

def build_llama_config(model_config: ModelConfig) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_token_id=model_config.pad_token_id,
        bos_token_id=model_config.bos_token_id,
        eos_token_id=model_config.eos_token_id,
        rope_theta=model_config.rope_theta,
        attention_dropout=model_config.dropout,
        use_cache=True,
    )


def load_model(
    weights_path: str,
    model_config: ModelConfig,
    device: torch.device,
    use_fp16: bool,
) -> PianoLLaMA:
    model = PianoLLaMA(build_llama_config(model_config))
    state_dict = safetensors.torch.load_file(weights_path)
    model.load_state_dict(state_dict, strict=True)

    if use_fp16:
        model = model.half()

    model.to(device)
    model.eval()
    return model


def build_tokenizer(config: ModelConfig) -> PianoRollTokenizer:
    return PianoRollTokenizer(
        patch_h=config.patch_h,
        patch_w=config.patch_w,
        marker_offset=81,
        measures_length=88,
        end_marker_part0=170,
        end_marker_part1=171,
        empty_marker=169,
        img_h=88,
    )


def pad_measure_tokens(
    tokens: Sequence[int],
    measure_len: int,
    eos_token_id: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, int]:
    tokens = list(tokens) + [eos_token_id]
    trimmed = tokens[:measure_len]
    pad_len = measure_len - len(trimmed)
    tokens = trimmed
    if pad_len > 0:
        tokens = tokens + [pad_token_id] * pad_len
    valid_length = len(trimmed)
    return torch.tensor(tokens, dtype=torch.long), valid_length


def clean_generated_tokens(
    tokens: List[int],
    stop_token: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> List[int]:
    cleaned: List[int] = []
    for token in tokens:
        if token == pad_token_id:
            break
        if token == bos_token_id and not cleaned:
            continue
        cleaned.append(token)
        if token == stop_token:
            break
    cleaned = [tok for tok in cleaned if tok != eos_token_id]
    if stop_token not in cleaned:
        cleaned.append(stop_token)
    return cleaned


def reconstruct_measure_tokens(
    model: PianoLLaMA,
    raw_tokens: Sequence[int],
    stop_token: int,
    measure_len: int,
    conditioning_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device: torch.device,
) -> torch.Tensor:
    measure_tensor, valid_length = pad_measure_tokens(
        raw_tokens,
        measure_len,
        model.config.eos_token_id,
        model.config.pad_token_id,
    )
    conditioning_length = min(
        max(conditioning_tokens, 0),
        valid_length,
    )
    result = model.reconstruct_measure(
        gt_tokens=measure_tensor.to(device),
        conditioning_length=conditioning_length,
        max_steps=measure_tensor.size(0),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_ids=[stop_token],
    )
    generated = result["all_tokens"].detach().cpu().tolist()
    cleaned = clean_generated_tokens(
        generated,
        stop_token=stop_token,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )
    return torch.tensor(cleaned, dtype=torch.long)


def reconstruct_piece(
    model: PianoLLaMA,
    tokenizer: PianoRollTokenizer,
    gt_path: str,
    measure_len: int,
    conditioning_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], dict]:
    save_dict = np.load(gt_path, allow_pickle=True)
    metadata = save_dict["metadata"].item()
    num_measures = metadata["num_measures"]

    part0_sequences: List[torch.Tensor] = []
    part1_sequences: List[torch.Tensor] = []

    for idx in range(num_measures):
        measure = save_dict[f"measure_{idx}"]
        part0_tokens, part1_tokens = encode_measure_tokens(measure, tokenizer)

        part0 = reconstruct_measure_tokens(
            model,
            part0_tokens,
            stop_token=256,
            measure_len=measure_len,
            conditioning_tokens=conditioning_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
        )
        part1 = reconstruct_measure_tokens(
            model,
            part1_tokens,
            stop_token=256,
            measure_len=measure_len,
            conditioning_tokens=conditioning_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
        )

        part0_sequences.append(part0)
        part1_sequences.append(part1)

        if (idx + 1) % 10 == 0 or idx + 1 == num_measures:
            print(f"Reconstructed measure {idx + 1}/{num_measures}")

    return part0_sequences, part1_sequences, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct every measure from GT and export MIDI."
    )
    parser.add_argument("--gt-path", required=True, help="Path to GT npz file.")
    parser.add_argument("--model-path", required=True, help="Path to model weights.")
    parser.add_argument(
        "--output-midi",
        default="reconstructed.mid",
        help="Destination MIDI path for reconstructed result.",
    )
    parser.add_argument(
        "--gt-midi",
        default="GT.mid",
        help="Optional path to also export the original GT as MIDI.",
    )
    parser.add_argument(
        "--measure-len",
        type=int,
        default=40,
        help="Fixed token length per measure (must match training).",
    )
    parser.add_argument(
        "--conditioning-tokens",
        type=int,
        default=1,
        help="Number of GT tokens to use as autoregressive prefix.",
    )
    parser.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 inference.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--velocity", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model_config = ModelConfig()
    tokenizer = build_tokenizer(model_config)
    model = load_model(args.model_path, model_config, device, args.fp16)
    print(args.output_midi)
    part0_sequences, part1_sequences, metadata = reconstruct_piece(
        model=model,
        tokenizer=tokenizer,
        gt_path=args.gt_path,
        measure_len=args.measure_len,
        conditioning_tokens=args.conditioning_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=1.0,
        device=device,
    )

    result = {
        "part0_beats": part0_sequences,
        "part1_beats": part1_sequences,
        "metadata": metadata,
        "GT_path": args.gt_path,
    }

    tempo = metadata.get("bpm", 120) or 120
    
    tokens_to_midi(
        result_dict=result,
        save_path=args.output_midi,
        velocity=args.velocity,
        tempo=tempo,
    )
    print(f"Saved reconstructed MIDI to {args.output_midi}")

    if args.gt_midi:
        save_gt_midi(
            save_path=args.gt_midi,
            gt_path=args.gt_path,
            velocity=args.velocity,
            dual_track=True,
        )
        print(f"Saved GT reference MIDI to {args.gt_midi}")


if __name__ == "__main__":
    main()
