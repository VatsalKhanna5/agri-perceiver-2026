import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

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

        # Special tokens
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        self.image_start_id = tokenizer.convert_tokens_to_ids("<image_start>")
        self.image_end_id = tokenizer.convert_tokens_to_ids("<image_end>")

    # --------------------------------------------------
    # 1. Visual Encoding Pipeline
    # --------------------------------------------------
    def encode_images(self, images):
        """
        images: [B, 5, 3, 384, 384]
        """
        feats = self.siglip(images)              # [B,5,729,1152]
        feats = self.tile_embed(feats)

        B, T, P, D = feats.shape
        feats = feats.view(B, T * P, D)          # flatten tiles

        feats = self.projector(feats)            # [B,3645,3072]
        latents = self.perceiver(feats)          # [B,128,3072]

        return latents

    # --------------------------------------------------
    # 2. Token Splicing into Phi-3
    # --------------------------------------------------
    def splice_and_forward(
        self,
        input_ids,
        attention_mask,
        visual_latents,
        labels=None
    ):
        B = input_ids.size(0)
        embed_layer = self.phi3.get_input_embeddings()

        new_embeds, new_masks, new_labels = [], [], []

        for b in range(B):
            ids = input_ids[b]
            mask = attention_mask[b]

            img_pos = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                raise ValueError("No <image> token found")

            pos = img_pos[0]

            before_ids = ids[:pos]
            after_ids = ids[pos+1:]

            before_embed = embed_layer(before_ids)
            after_embed = embed_layer(after_ids)

            start_embed = embed_layer(
                torch.tensor([self.image_start_id], device=ids.device)
            )
            end_embed = embed_layer(
                torch.tensor([self.image_end_id], device=ids.device)
            )

            visual = visual_latents[b]

            seq_embed = torch.cat([
                before_embed,
                start_embed,
                visual,
                end_embed,
                after_embed
            ], dim=0)

            new_embeds.append(seq_embed)
            new_masks.append(torch.ones(seq_embed.size(0), device=ids.device))

            if labels is not None:
                lab = labels[b]
                before_lab = lab[:pos]
                after_lab = lab[pos+1:]

                ignore = torch.full(
                    (1 + visual.size(0) + 1,),
                    -100,
                    device=ids.device
                )

                new_lab = torch.cat([before_lab, ignore, after_lab], dim=0)
                new_labels.append(new_lab)

        new_embeds = pad_sequence(new_embeds, batch_first=True)
        new_masks = pad_sequence(new_masks, batch_first=True)
        new_labels = pad_sequence(new_labels, batch_first=True, padding_value=-100)

        return self.phi3(
            inputs_embeds=new_embeds,
            attention_mask=new_masks,
            labels=new_labels
        )

    # --------------------------------------------------
    # 3. Full Forward Pass
    # --------------------------------------------------
    def forward(self, input_ids, attention_mask, images, labels=None):
        visual_latents = self.encode_images(images)
        outputs = self.splice_and_forward(
            input_ids,
            attention_mask,
            visual_latents,
            labels
        )
        return outputs
