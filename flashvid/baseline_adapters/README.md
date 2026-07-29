# LLaVA baseline adapters

These adapters keep each released method's algorithmic structure while wiring it
to the SigLIP/Qwen2 layout used by LLaVA-OneVision and LLaVA-Video.

- `fastv` ([upstream](https://github.com/pkunlp-icler/FastV)): unchanged
  visual tokens followed by FastV attention pruning at the
  configured language layer.
- `fastvid` ([upstream](https://github.com/LunarShen/FastVID)): FastVID DySeg,
  STPrune, and DTM with the released SigLIP
  pooling-head attention and frame descriptors.
- `visionzip` ([upstream](https://github.com/JIA-Lab-research/VisionZip)):
  VisionZip dominant/contextual selection using the vision-layer
  attention and key metric. LLaVA SigLIP has no CLS token, so its released
  Qwen-style all-query attention score is used.
- `prunevid` ([upstream](https://github.com/Visual-AI/PruneVid)): PruneVID
  temporal/static/dynamic DPC merging and group-wise text-attention pruning.
  This is a backbone adapter because the public upstream implementation is tied
  to PLLaVA rather than LLaVA-OneVision.

The runners disable expansion and unrelated inner pruning for the outer-only
baselines. FastV uses its own target retention directly. PruneVID computes the
inner ratio needed to reach the requested final token budget after its visual
merge.

The adapters are model-specific thin patches rather than Qwen3 emulations. The
runner layout follows the same engineering principle as
[VidCom2](https://github.com/xuyang-liu16/VidCom2): keep the released selection
operator and adapt only the model-specific feature and attention plumbing.
