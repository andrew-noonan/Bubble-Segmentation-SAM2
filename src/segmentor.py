from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.config import CFG_PATH, CKPT_PATH

def load_predictor():
    # Override device explicitly using Hydra override string
    model = build_sam2(CFG_PATH, CKPT_PATH, overrides=["model.device=cpu"])
    model = model.to("cpu").eval()
    return SAM2ImagePredictor(model)



def run_sam_on_image(predictor, image_rgb):
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=DTYPE):
        predictor.set_image(image_rgb)
        masks, _, _ = predictor.predict(
            point_coords=None, point_labels=None, box=None
        )
    return masks
