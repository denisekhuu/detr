# ------------------------------------------------------------------------
# Copyright 2026 Denise-Phi Khuu. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .detr import build

def build_model(args):
    print("Building Hydra DETR model...")
    return build(args)
