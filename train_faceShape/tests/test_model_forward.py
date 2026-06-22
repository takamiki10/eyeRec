import torch

from train_face_shape.model import build_model


def test_model_forward_pass_with_dummy_tensor():
    model = build_model(num_classes=5, use_pretrained=False)
    model.eval()

    dummy_images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        outputs = model(dummy_images)

    assert outputs.shape == (2, 5)
