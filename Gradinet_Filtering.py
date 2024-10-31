import torch
import cv2
import numpy as np
from typing import List
from pytorch_grad_cam.base_cam import BaseCAM


class GradCAM_Logits(BaseCAM):
    def __init__(self, model, target_layers, reshape_transform=None, target_cls=0):
        super(GradCAM_Logits, self).__init__(model, target_layers, reshape_transform, target_cls)
        self.target_cls = target_cls

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # 2D image
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))

        # 3D image
        elif len(grads.shape) == 5:
            return np.mean(grads, axis=(2, 3, 4))

        else:
            raise ValueError("Invalid grads shape."
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")

    def forward(self, input_tensor: torch.Tensor, targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)

        result = np.argmax(outputs.cpu().data.numpy(), axis=-1)[0]

        if result == self.target_cls:
            if self.uses_gradients:
                self.model.zero_grad()
                loss = sum([target(output) for target, output in zip(targets, outputs)])
                loss.backward(retain_graph=True)

            cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)

            return self.aggregate_multi_layers(cam_per_layer), result
        else:
            return None, result


def Gradient_Filtering(cam, g_th):
    """
    :param cam:
    :param g_th:
    :return: [[xmin, ymin, xmax, ymax],...]
    """
    cam_min = cam.min(axis=(1, 2), keepdims=True)  # [N, H, W] -> [N, 1, 1]
    cam_max = cam.max(axis=(1, 2), keepdims=True)  # [N, H, W] -> [N, 1, 1]
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)  # [N, H, W]
    thresh = np.where(cam >= g_th, 255, 0).astype(np.uint8)

    results = []

    for im in thresh:
        bboxes = []
        counters, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(counters):
            for cnt in counters:
                bbox = xywh2xyxy(cv2.boundingRect(cnt))
                bboxes.append(bbox.tolist())
        else:
            bboxes.append([])

        results.append(bboxes)

    return results


def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x).astype(np.float)
    y[0] = (x[0])
    y[1] = (x[1])
    y[2] = (x[0] + x[2])
    y[3] = (x[1] + x[3])
    return y
