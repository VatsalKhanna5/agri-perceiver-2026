import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.spatial.tile_embeddings import TileEmbeddings
from models.projector.vision_projector import VisionProjector
from models.perceiver.perceiver_resampler import PerceiverResampler

class AgriPerceiverVLM(nn.Module):
    def __init__(self, siglip_model, phi3_model, tokenizer):
        super().__init__()

        self.siglip = siglip_model
        self.phi3 = phi3_model
        self.tokenizer = tokenizer

        # Vision stack
        self.tile_embed = TileEmbeddings()
        self.projector = VisionProjector()
        self.perceiver = PerceiverResampler()

        # Force connector modules to match LLM precision (bfloat16)
        self.tile_embed.to(torch.bfloat16)
        self.projector.to(torch.bfloat16)
        self.perceiver.to(torch.bfloat16)

        # Special tokens
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        self.image_start_id = tokenizer.convert_tokens_to_ids("<image_start>")
        self.image_end_id = tokenizer.convert_tokens_to_ids("<image_end>")

    def encode_images(self, pixel_values):
        """
        pixel_values: [B, 5, 3, 384, 384]
        """
        B, T, C, H, W = pixel_values.shape
        pixel_values = pixel_values.to(dtype=torch.bfloat16)
        
        # 1. Flatten for SigLIP vision backbone
        pixel_values = pixel_values.view(B * T, C, H, W)
        siglip_outputs = self.siglip.vision_model(pixel_values=pixel_values)
        feats = siglip_outputs.last_hidden_state.to(dtype=torch.bfloat16)
        
        # 2. Spatial grounding via TileEmbeddings
        feats = feats.view(B, T, -1, feats.size(-1))
        feats = self.tile_embed(feats)
        
        # 3. Projection and Resampling (Compression to 128 latents)
        feats = feats.reshape(B, -1, feats.size(-1)) 
        feats = self.projector(feats)
        latents = self.perceiver(feats)
        
        return latents.to(dtype=torch.bfloat16)

    def splice_and_forward(self, input_ids, attention_mask, visual_latents, labels=None):
        B = input_ids.size(0)
        embed_layer = self.phi3.get_input_embeddings()
        target_dtype = self.phi3.dtype
        device = input_ids.device

        new_embeds, new_masks, new_labels = [], [], []

        for b in range(B):
            ids = input_ids[b]
            img_pos = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                raise ValueError("No <image> token found in input_ids")

            pos = img_pos[0]
            
            # Text embeddings before and after the <image> placeholder
            before_embed = embed_layer(ids[:pos]).to(target_dtype)
            after_embed = embed_layer(ids[pos+1:]).to(target_dtype)

            # Splicing visual boundaries (<image_start> and <image_end>)
            start_embed = embed_layer(torch.tensor([self.image_start_id], device=device)).to(target_dtype)
            end_embed = embed_layer(torch.tensor([self.image_end_id], device=device)).to(target_dtype)

            visual = visual_latents[b].to(target_dtype)

            # Construct the hybrid sequence
            seq_embed = torch.cat([before_embed, start_embed, visual, end_embed, after_embed], dim=0)
            new_embeds.append(seq_embed)
            
            # Attention mask for the new sequence length
            new_masks.append(torch.ones(seq_embed.size(0), device=device, dtype=torch.long))

            if labels is not None:
                lab = labels[b]
                # Image tokens and boundaries are ignored in loss (-100)
                # 1 (start) + N (latents) + 1 (end)
                ignore = torch.full((1 + visual.size(0) + 1,), -100, device=device)
                new_lab = torch.cat([lab[:pos], ignore, lab[pos+1:]], dim=0)
                new_labels.append(new_lab)

        # Pad constructed sequences for batching
        new_embeds = pad_sequence(new_embeds, batch_first=True)
        new_masks = pad_sequence(new_masks, batch_first=True)
        
        final_labels = None
        if labels is not None and len(new_labels) > 0:
            final_labels = pad_sequence(new_labels, batch_first=True, padding_value=-100)

        # CRITICAL: use_cache=False bypasses the DynamicCache AttributeError in Phi-3
        outputs = self.phi3(
            inputs_embeds=new_embeds,
            attention_mask=new_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False 
        )

        # Reconstruct the output object with our spliced labels attached
        return CausalLMOutputWithPast(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), final_labels

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        # Sync input to backbone precision
        target_dtype = self.phi3.dtype
        pixel_values = pixel_values.to(target_dtype)

        # Visual pipeline
        visual_latents = self.encode_images(pixel_values)
        visual_latents = visual_latents.to(target_dtype)

        return self.splice_and_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_latents=visual_latents,
            labels=labels
        )