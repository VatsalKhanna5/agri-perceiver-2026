"""
AgriPerceiver VLM — Main Vision-Language Model.

Architecture: SigLIP (vision) → TileEmbeddings → VisionProjector → PerceiverResampler → Phi-3 (language)
The visual bridge (TileEmbed + Projector + Perceiver) compresses 5×729 = 3,645 patch tokens into 128 latents.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast

from agri_perceiver.model.tile_embeddings import TileEmbeddings
from agri_perceiver.model.vision_projector import VisionProjector
from agri_perceiver.model.perceiver_resampler import PerceiverResampler


class AgriPerceiverVLM(nn.Module):
    """
    Perceiver-based Vision-Language Model for agricultural pathology diagnosis.

    The architecture is backbone-agnostic: any vision encoder producing [B, T, P, D_v]
    patch tokens can be paired with any causal LM accepting [B, S, D_l] embeddings.
    The Perceiver bridge handles the cross-modal alignment and token compression.

    Args:
        vision_encoder: Vision backbone (e.g., SigLIP). Must expose `.vision_model(pixel_values=...)`.
        language_model: Causal LM (e.g., Phi-3). Must expose `.get_input_embeddings()`.
        tokenizer: Tokenizer with special tokens <image>, <image_start>, <image_end>.
        vision_dim: Vision encoder output dimension (default: 1152 for SigLIP-SO400M).
        language_dim: Language model hidden dimension (default: 3072 for Phi-3-mini).
        num_latents: Number of Perceiver latent vectors (default: 128).
        perceiver_depth: Number of Perceiver blocks (default: 2).
        perceiver_heads: Number of attention heads in Perceiver (default: 24).
        num_tiles: Number of AnyRes tiles (default: 5 = 4 quadrants + 1 global).
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        language_model: nn.Module,
        tokenizer,
        vision_dim: int = 1152,
        language_dim: int = 3072,
        num_latents: int = 128,
        perceiver_depth: int = 2,
        perceiver_heads: int = 24,
        num_tiles: int = 5,
    ):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.tokenizer = tokenizer

        # Visual bridge: TileEmbed → Projector → Perceiver
        self.tile_embed = TileEmbeddings(dim=vision_dim, num_tiles=num_tiles)
        self.projector = VisionProjector(in_dim=vision_dim, out_dim=language_dim)
        self.perceiver = PerceiverResampler(
            dim=language_dim,
            num_latents=num_latents,
            depth=perceiver_depth,
            heads=perceiver_heads,
        )

        # Special token IDs
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        self.image_start_id = tokenizer.convert_tokens_to_ids("<image_start>")
        self.image_end_id = tokenizer.convert_tokens_to_ids("<image_end>")

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode tiled images through the full visual pipeline.

        Args:
            pixel_values: [B, T, C, H, W] tiled images (T=num_tiles).

        Returns:
            [B, num_latents, language_dim] compressed visual latents.
        """
        B, T, C, H, W = pixel_values.shape
        target_dtype = next(self.projector.parameters()).dtype

        # 1. Flatten tiles for batch processing through vision encoder
        flat = pixel_values.view(B * T, C, H, W).to(dtype=target_dtype)
        vision_out = self.vision_encoder.vision_model(pixel_values=flat)
        feats = vision_out.last_hidden_state.to(dtype=target_dtype)  # [B*T, P, D_v]

        # 2. Reshape and apply spatial grounding
        feats = feats.view(B, T, -1, feats.size(-1))  # [B, T, P, D_v]
        feats = self.tile_embed(feats)

        # 3. Project to language dimension and compress via Perceiver
        feats = feats.reshape(B, -1, feats.size(-1))  # [B, T*P, D_v]
        feats = self.projector(feats)                   # [B, T*P, D_l]
        latents = self.perceiver(feats)                  # [B, num_latents, D_l]

        return latents.to(dtype=target_dtype)

    def splice_and_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_latents: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        Replace the <image> placeholder with visual latents and forward through the LLM.

        Constructs: [text_before | <image_start> | visual_latents | <image_end> | text_after]

        Args:
            input_ids: [B, S] token IDs containing <image> placeholder.
            attention_mask: [B, S] attention mask.
            visual_latents: [B, N, D] compressed visual features.
            labels: [B, S] optional training labels.

        Returns:
            Tuple of (CausalLMOutputWithPast, adjusted_labels_or_None).
        """
        B = input_ids.size(0)
        embed_layer = self.language_model.get_input_embeddings()
        target_dtype = next(self.language_model.parameters()).dtype
        device = input_ids.device

        new_embeds, new_masks, new_labels = [], [], []

        for b in range(B):
            ids = input_ids[b]
            img_pos = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                raise ValueError("No <image> token found in input_ids")

            pos = img_pos[0]

            # Text embeddings around the placeholder
            before_embed = embed_layer(ids[:pos]).to(target_dtype)
            after_embed = embed_layer(ids[pos + 1 :]).to(target_dtype)

            # Boundary markers
            start_embed = embed_layer(torch.tensor([self.image_start_id], device=device)).to(target_dtype)
            end_embed = embed_layer(torch.tensor([self.image_end_id], device=device)).to(target_dtype)

            visual = visual_latents[b].to(target_dtype)

            # Construct hybrid sequence
            seq_embed = torch.cat([before_embed, start_embed, visual, end_embed, after_embed], dim=0)
            new_embeds.append(seq_embed)
            new_masks.append(torch.ones(seq_embed.size(0), device=device, dtype=torch.long))

            if labels is not None:
                lab = labels[b]
                ignore = torch.full((1 + visual.size(0) + 1,), -100, device=device)
                new_lab = torch.cat([lab[:pos], ignore, lab[pos + 1 :]], dim=0)
                new_labels.append(new_lab)

        # Pad for batching
        new_embeds = pad_sequence(new_embeds, batch_first=True)
        new_masks = pad_sequence(new_masks, batch_first=True)

        final_labels = None
        if labels is not None and len(new_labels) > 0:
            final_labels = pad_sequence(new_labels, batch_first=True, padding_value=-100)

        outputs = self.language_model(
            inputs_embeds=new_embeds,
            attention_mask=new_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        return CausalLMOutputWithPast(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), final_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """Full forward: encode images → splice → LLM forward."""
        target_dtype = next(self.language_model.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)

        visual_latents = self.encode_images(pixel_values).to(target_dtype)

        return self.splice_and_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_latents=visual_latents,
            labels=labels,
        )

    @property
    def bridge_state_dict(self) -> dict:
        """Return only the trainable visual bridge weights (for Stage 1 checkpointing)."""
        return {
            "tile_embed": self.tile_embed.state_dict(),
            "projector": self.projector.state_dict(),
            "perceiver": self.perceiver.state_dict(),
        }

    def load_bridge_weights(self, path_or_dict, device="cpu"):
        """Load visual bridge weights from a path or dict."""
        if isinstance(path_or_dict, (str, bytes)):
            state = torch.load(path_or_dict, map_location=device, weights_only=True)
        else:
            state = path_or_dict
        self.tile_embed.load_state_dict(state["tile_embed"])
        self.projector.load_state_dict(state["projector"])
        self.perceiver.load_state_dict(state["perceiver"])
