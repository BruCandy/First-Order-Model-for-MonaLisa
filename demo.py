import sys
import yaml

import numpy as np
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.full_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector

class AnimationMaker:
    def __init__(self, generator, kp_detector, source_image, relative=True, adapt_movement_scale=True, cpu=False):
        self.generator = generator
        self.kp_detector = kp_detector
        self.relative = relative
        self.adapt_movement_scale = adapt_movement_scale
        self.cpu = cpu
        self.kp_driving_initial = None

        # ソース画像の処理
        self.source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            self.source = self.source.cuda()

        # ソース画像のキーポイントを検出
        self.kp_source = kp_detector(self.source)

    def make_animation(self, driving_frame):
        with torch.no_grad():
            driving_frame_tensor = torch.tensor(driving_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not self.cpu:
                driving_frame_tensor = driving_frame_tensor.cuda()

            if self.kp_driving_initial is None:
                self.kp_driving_initial = self.kp_detector(driving_frame_tensor)

            kp_driving = self.kp_detector(driving_frame_tensor)
            kp_norm = normalize_kp(kp_source=self.kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=self.kp_driving_initial, use_relative_movement=self.relative,
                                   use_relative_jacobian=self.relative, adapt_movement_scale=self.adapt_movement_scale)
            out = self.generator(self.source, kp_source=self.kp_source, kp_driving=kp_norm)
            prediction = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            return prediction
