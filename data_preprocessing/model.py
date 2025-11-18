import torch
import torch.nn as nn
from typing import Optional, Sequence

from transformers import LlamaForCausalLM, PreTrainedModel, LlamaModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class PianoLLaMA(PreTrainedModel):
    """Encoder-decoder style measure model built from two GPT blocks."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # encoder/decoder share the same configuration
        self.encoder = LlamaModel(self.config)
        self.decoder = LlamaForCausalLM(self.config)

        self.pad_token_id = config.pad_token_id
        self.hidden_size = config.hidden_size

        # match previous behaviour by re-initialising weights
        self.encoder.apply(self._init_weights)
        self.decoder.apply(self._init_weights)


    def _init_weights(self, module):
        """Initialise Linear/Embedding layers with the configured range."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            input_ids: (batch, num_measures, measure_len)
            attention_mask: 同shape，pad位置为0
            labels: (batch, num_measures, measure_len)，忽略位置填-100
        """
        if input_ids.dim() != 3:
            raise ValueError(
                f"Expected 3D measure tensor (batch, measures, step), but got shape {tuple(input_ids.shape)}"
            )

        batch_size, num_measures, measure_len = input_ids.shape
        flat_tokens = input_ids.reshape(batch_size * num_measures, measure_len)

        if attention_mask is not None:
            flat_attention = attention_mask.reshape(batch_size * num_measures, measure_len)
        else:
            flat_attention = None

        if labels is not None:
            flat_labels = labels.reshape(batch_size * num_measures, measure_len)
        else:
            flat_labels = None

        # ---------------------- encode ----------------------
        encoder_outputs = self.encoder(
            input_ids=flat_tokens,
            attention_mask=flat_attention,
            use_cache=False,
            output_hidden_states=False,
        )
        encoder_hidden = encoder_outputs.last_hidden_state  # (B*N, measure_len, hidden)
        measure_bos = encoder_hidden[:, -1, :].unsqueeze(1)  # (B*N, 1, hidden)

        # ---------------------- decode ----------------------
        token_embeds = self.decoder.model.embed_tokens(flat_tokens)  # (B*N, measure_len, hidden)
        decoder_inputs_embeds = torch.cat([measure_bos, token_embeds], dim=1)  # +1 for BOS

        if flat_attention is not None:
            bos_mask = torch.ones(
                (flat_attention.size(0), 1),
                dtype=flat_attention.dtype,
                device=input_ids.device,
            )
            decoder_attention = torch.cat([bos_mask, flat_attention], dim=1)
        else:
            decoder_attention = None

        if flat_labels is not None:
            bos_labels = torch.full(
                (flat_labels.size(0), 1),
                -100,
                dtype=flat_labels.dtype,
                device=input_ids.device,
            )
            decoder_labels = torch.cat([bos_labels, flat_labels], dim=1)
        else:
            decoder_labels = None

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention,
            labels=decoder_labels,
            use_cache=False if decoder_labels is not None else True,
        )

        # Drop BOS position so external consumers see logits aligned to step dimension
        logits = decoder_outputs.logits[:, 1:, :]
        logits = logits.reshape(batch_size, num_measures, measure_len, -1)

        loss = decoder_outputs.loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None
        )

    def _prepare_measure_bos(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )
        return encoder_outputs.last_hidden_state[:, -1:, :]

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= logits.size(-1):
            return logits
        top_values = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        mask = logits < top_values
        logits = logits.masked_fill(mask, float('-inf'))
        return logits

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p <= 0.0 or top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float('-inf'))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        return logits

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated: Sequence[int],
        penalty: float,
    ) -> torch.Tensor:
        if penalty == 1.0 or not generated:
            return logits
        unique_tokens = set(generated)
        for token_id in unique_tokens:
            token_id = int(token_id)
            token_logits = logits[..., token_id]
            logits[..., token_id] = torch.where(
                token_logits < 0,
                token_logits * penalty,
                token_logits / penalty,
            )
        return logits

    @torch.no_grad()
    def reconstruct_measure(
        self,
        gt_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        conditioning_length: Optional[int] = None,
        max_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_ids: Optional[Sequence[int]] = None,
    ) -> dict:
        """Autoregressively reconstruct a single measure conditioned on GT tokens."""

        if gt_tokens.dim() == 1:
            gt_tokens = gt_tokens.unsqueeze(0)
        elif gt_tokens.dim() != 2 or gt_tokens.size(0) != 1:
            raise ValueError("gt_tokens must have shape (seq,) or (1, seq)")

        device = gt_tokens.device



        max_steps = max_steps or gt_tokens.size(1)
        max_steps = min(max_steps, gt_tokens.size(1))

        stop_id_set = set(stop_ids) if stop_ids is not None else set()
        stop_id_set.update({self.config.eos_token_id, self.pad_token_id})

        measure_bos = self._prepare_measure_bos(gt_tokens, attention_mask)
        generated_tokens = []

        while len(generated_tokens) < max_steps:
            if len(generated_tokens) > 0:
                decoder_input = torch.tensor(generated_tokens, device=device).unsqueeze(0)
                token_embeds = self.decoder.model.embed_tokens(decoder_input)
                decoder_inputs_embeds = torch.cat([measure_bos, token_embeds], dim=1)
            else:
                decoder_inputs_embeds = measure_bos

            decoder_outputs = self.decoder(
                inputs_embeds=decoder_inputs_embeds,
                use_cache=False,
            )
            next_token_logits = decoder_outputs.logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(next_token_logits, dim=-1)
            else:
                scaled_logits = next_token_logits / temperature
                scaled_logits = self._apply_repetition_penalty(
                    scaled_logits, generated_tokens, repetition_penalty
                )
                scaled_logits = self._apply_top_k(scaled_logits, top_k)
                scaled_logits = self._apply_top_p(scaled_logits, top_p)
                probs = torch.softmax(scaled_logits, dim=-1)
                if torch.isnan(probs).any():
                    next_token = torch.argmax(next_token_logits, dim=-1)
                else:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            next_id = int(next_token.item())
            generated_tokens.append(next_id)

            if next_id in stop_id_set:
                print(f"Stopping{next_id} at step {len(generated_tokens)}")
                break

        generated_tensor = torch.tensor(generated_tokens, device=device)
        return {
            "conditioning_length": conditioning_length,
            "all_tokens": generated_tensor,
            "generated_tail": generated_tensor[conditioning_length:],
        }

