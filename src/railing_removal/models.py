from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image


CLIPSEG_MODEL_ID = "CIDAS/clipseg-rd64-refined"
SAM2_MODEL_ID = "facebook/sam2-hiera-tiny"


class ClipSegPredictor:
    def __init__(
        self,
        model_id: str = CLIPSEG_MODEL_ID,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = CLIPSegProcessor.from_pretrained(
            model_id,
            use_fast=False,
        )
        self._model = (
            CLIPSegForImageSegmentation.from_pretrained(model_id)
            .eval()
            .to(self._device)
        )

    def __call__(
        self,
        image: Image.Image,
        prompts: tuple[str, ...],
    ) -> np.ndarray:
        inputs = self._processor(
            text=list(prompts),
            images=[image] * len(prompts),
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits
        resized = self._torch.nn.functional.interpolate(
            logits[:, None],
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        return self._torch.sigmoid(resized).cpu().numpy()


class Sam2Predictor:
    def __init__(
        self,
        model_id: str = SAM2_MODEL_ID,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = Sam2Processor.from_pretrained(model_id)
        self._model = Sam2Model.from_pretrained(model_id).eval().to(self._device)

    def __call__(
        self,
        image: Image.Image,
        points: list[list[float]],
        labels: list[int],
    ) -> tuple[np.ndarray, float]:
        inputs = self._processor(
            images=image,
            input_points=[[points]],
            input_labels=[[labels]],
            return_tensors="pt",
        ).to(self._device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0][0]
        scores = outputs.iou_scores[0, 0].detach().cpu().numpy()
        best = int(np.argmax(scores))
        return masks[best].numpy().astype(bool), float(scores[best])
