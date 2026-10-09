"""Explicit parameter-name mapping for the supplied CvT/LCvT state dictionary."""

import re


def source_key(key):
    match = re.fullmatch(r"stages\.(\d+)\.(.+)", key)
    if match:
        stage, rest = int(match[1]) + 1, match[2]
        if rest.startswith("embedding."):
            return f"stage{stage}_conv_embed.0." + rest.removeprefix("embedding.")
        if rest.startswith("norm."):
            return f"stage{stage}_conv_embed.2." + rest.removeprefix("norm.")
        block, rest = re.fullmatch(r"blocks\.(\d+)\.(.+)", rest).groups()
        prefix = f"stage{stage}_transformer.0.layers.{block}."
        for new, old in (("attention_norm.", "0.norm."), ("mlp_norm.", "1.norm."),
                         ("mlp.", "1.fn.net."), ("attention.output.", "0.fn.to_out.")):
            if rest.startswith(new):
                return prefix + old + rest.removeprefix(new)
        projection, part, suffix = re.fullmatch(r"attention\.(query|key|value)\.(\d+)\.(.+)", rest).groups()
        return prefix + f"0.fn.to_{dict(query='q',key='k',value='v')[projection]}." + (
            ("depthwise", "bn", "pointwise")[int(part)] + "." + suffix
        )
    match = re.fullmatch(r"branches\.(\d+)\.(.+)", key)
    if match:
        lod, rest = int(match[1]) + 1, match[2]
        prefix = f"LoD{lod}_"
        for new, old in (("embedding.", "patch_embed_just.projection."), ("head.", "head.")):
            if rest.startswith(new):
                return prefix + old + rest.removeprefix(new)
        if rest in ("cls_token", "coarse_position", "fine_position"):
            return prefix + dict(cls_token="cls_token", coarse_position="pos_emb", fine_position="pos_emb_fine")[rest]
        if rest.startswith("reuse."):
            block = "reuse_block" if lod == 1 else "reuse_block2"
            part, suffix = rest.removeprefix("reuse.").split(".", 1)
            return block + "." + {"0":"0", "1":"1.fc1", "4":"1.fc2"}[part] + "." + suffix
        mode, index, rest = re.fullmatch(r"(coarse|fine)_blocks\.(\d+)\.(.+)", rest).groups()
        prefix += f"blocks_{'c' if mode == 'coarse' else 'f'}.{index}."
        for new, old in (("attention_norm.", "MSA.norm."), ("qkv.", "MSA.linear_qkv."), ("mlp.", "FFN.block.")):
            if rest.startswith(new):
                return prefix + old + rest.removeprefix(new)
    raise ValueError(f"No source mapping for parameter {key}")


def import_source_state(model, source_state):
    """Load every required tensor, rejecting missing/shape-incompatible keys."""
    target = {}
    used = set()
    for key, tensor in model.state_dict().items():
        original = source_key(key)
        if original not in source_state or source_state[original].shape != tensor.shape:
            raise ValueError(f"Missing or incompatible source tensor: {original} -> {key}")
        target[key] = source_state[original]
        used.add(original)
    ignored = sorted(set(source_state) - used)
    allowed = re.compile(r"LoD[12]_patch_embed_conv\..+|LoD2_pos_emb_fine|LoD[12]_blocks_[cf]\.\d+\.FFN\.(norm|fc1|fc2)\..+")
    unexpected = [key for key in ignored if not allowed.fullmatch(key)]
    if unexpected:
        raise ValueError(f"Unexpected source tensors: {unexpected}")
    model.load_state_dict(target, strict=True)
    return ignored
