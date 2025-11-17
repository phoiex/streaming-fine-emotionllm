from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn


SPECIAL_TOKENS = [
    "<VIDEO>",
    "</VIDEO>",
    "<AUDIO>",
    "</AUDIO>",
    "<TEXT>",
    "</TEXT>",
    "<EMO>",
]


def ensure_special_tokens(tokenizer, model: nn.Module):
    to_add = [t for t in SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))


@dataclass
class PromptPieces:
    system: str = "You are an assistant that performs emotion recognition from conversation."
    instruction: str = (
        "Given the following video features, audio content, and dialogue context, "
        "predict the speaker's emotion."
    )


def assemble_inputs_with_pseudotokens(
    tokenizer,
    lm: nn.Module,
    h_v: torch.Tensor,  # (B, H)
    h_a: torch.Tensor,  # (B, H)
    asr_texts: List[str],
    pieces: PromptPieces = PromptPieces(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build inputs_embeds by inserting pseudo-token embeddings for h_v and h_a.

    Returns:
        inputs_embeds: (B, L, H)
        attention_mask: (B, L)
        emo_positions: (B,) index of <EMO> token per sample
    """
    device = next(lm.parameters()).device
    embed = lm.get_input_embeddings()
    hidden_size = embed.weight.shape[1]

    assert h_v.shape == h_a.shape and h_v.ndim == 2
    B = h_v.shape[0]
    hv = h_v.to(device).unsqueeze(1)  # (B,1,H)
    ha = h_a.to(device).unsqueeze(1)
    if hv.shape[-1] != hidden_size:
        raise ValueError("Projection dim mismatch: h_v does not match LM hidden size")
    if ha.shape[-1] != hidden_size:
        raise ValueError("Projection dim mismatch: h_a does not match LM hidden size")

    # Prepare fixed text scaffolding around pseudo tokens
    fixed_prefix = (
        f"<s> {pieces.system}\n{pieces.instruction} \n<VIDEO>"
    )
    fixed_between = "</VIDEO> <AUDIO>"
    fixed_mid = "</AUDIO> <TEXT>"
    fixed_post = "</TEXT> <EMO>"

    inputs_embeds: List[torch.Tensor] = []
    attn_masks: List[torch.Tensor] = []
    emo_positions: List[int] = []

    for i in range(B):
        text_i = asr_texts[i]
        # Tokenize each part to get token embeddings
        ids_prefix = tokenizer(fixed_prefix, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(device)
        ids_between = tokenizer(
            fixed_between, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)
        ids_mid = tokenizer(fixed_mid, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(device)
        ids_asr = tokenizer(text_i, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(device)
        ids_post = tokenizer(fixed_post, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(device)

        # Convert tokens to embeddings
        e_prefix = embed(ids_prefix)[0]
        e_between = embed(ids_between)[0]
        e_mid = embed(ids_mid)[0]
        e_asr = embed(ids_asr)[0]
        e_post = embed(ids_post)[0]

        # Construct sequence: prefix, hv, between, ha, mid, asr, post
        seq = torch.cat(
            [e_prefix, hv[i], e_between, ha[i], e_mid, e_asr, e_post], dim=0
        )  # (L_i, H)

        # Compute <EMO> index (assumes last token in e_post is <EMO>)
        emo_pos = seq.shape[0] - 1  # last token
        inputs_embeds.append(seq)
        attn_masks.append(torch.ones(seq.shape[0], device=device, dtype=torch.long))
        emo_positions.append(emo_pos)

    # Pad to max length
    max_len = max(x.shape[0] for x in inputs_embeds)
    padded = []
    masks = []
    for seq, mask in zip(inputs_embeds, attn_masks):
        if seq.shape[0] < max_len:
            pad = torch.zeros((max_len - seq.shape[0], hidden_size), device=device, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=0)
            padm = torch.zeros((max_len - mask.shape[0],), device=device, dtype=mask.dtype)
            mask = torch.cat([mask, padm], dim=0)
        padded.append(seq.unsqueeze(0))
        masks.append(mask.unsqueeze(0))

    inputs_embeds_b = torch.cat(padded, dim=0)
    attention_mask_b = torch.cat(masks, dim=0)
    emo_positions_b = torch.tensor(emo_positions, device=device, dtype=torch.long)
    return inputs_embeds_b, attention_mask_b, emo_positions_b

