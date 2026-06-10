# Component Reference

This page summarizes the components that ship with VCR-Bench. Use the command-line names in the `CLI name` column with `--model`, `--attack`, `--defence`, and preset files under `configs/`.

## Models

VCR-Bench model wrappers normalize each architecture behind the same classifier interface: raw videos are decoded to `NTHWC`, sampled according to the active pipeline stage, transformed by model-specific preprocessing, and evaluated with clip aggregation.

| CLI name | Display name | Default preset | Description |
|---|---|---|---|
| `actionclip` | ActionCLIP | `vit_b16_kinetics400` | CLIP-style video action recognition model with a ViT-B/16 backbone and Kinetics-400 weights. |
| `amd` | AMD | `vitb_k400` | Asymmetric Masked Distillation video classifier using a ViT-B backbone. |
| `c2d` | C2D | `r50_kinetics400` | 2D convolutional ResNet-style baseline applied to video clips. |
| `c3d` | C3D | `c3d_ucf101` | Classical 3D convolutional network for action recognition. |
| `csn` | CSN | `r50_kinetics400` | Channel-separated 3D convolutional network with a ResNet-50 backbone. |
| `i3d` | I3D | `r50_kinetics400` | Inflated 3D convolutional action-recognition model. |
| `i3dnonlocal` | I3D Non-Local | `r50_kinetics400` | I3D variant with non-local blocks for longer-range spatiotemporal context. |
| `ila` | ILA / X-CLIP wrapper | `vit_b16_kinetics400` | Multimodal video classifier built around the integrated ILA/X-CLIP implementation. |
| `internvideo` | InternVideo | `vit_base_p16_kinetics400` | Video foundation model wrapper using a ViT-Base patch-16 backbone. |
| `internvideo2` | InternVideo2 | `vit_1b_p14_kinetics400` | Larger InternVideo2 foundation model wrapper with ViT-1B patch-14 weights. |
| `mvitv2` | MViTv2 | `base_p244_kinetics400` | Multiscale Vision Transformer v2 video classifier. |
| `onepeace` | ONE-PEACE | `vit_l40_kinetics400` | Large multimodal foundation model adapted for video classification. |
| `r2plus1d` | R(2+1)D | `r34_kinetics400` | Factorized 3D convolutional network that separates spatial and temporal convolutions. |
| `slowfast` | SlowFast | `r101_kinetics400` | Dual-pathway video model with slow semantic and fast motion streams. |
| `slowonly` | SlowOnly | `r50_kinetics400` | Single-pathway SlowFast-family baseline with a ResNet-50 backbone. |
| `tadaformer` | TAdaFormer | `large_14_kinetics400` | Temporally adaptive transformer model from the TAdaConv/TAdaFormer family. |
| `tanet` | TANet | `r50_kinetics400` | Temporal Adaptive Network for video recognition. |
| `timesformer` | TimeSformer | `vit_base_p16_kinetics400` | Pure transformer video classifier based on divided space-time attention. |
| `tin` | TIN | `r50_sthv2` | Temporal Interlacing Network wrapper for Something-Something-v2 style recognition. |
| `tpn` | TPN | `r50_kinetics400` | Temporal Pyramid Network for multi-scale temporal modeling. |
| `trn` | TRN | `r50_sthv2` | Temporal Relation Network wrapper for relational temporal reasoning. |
| `tsm` | TSM | `r50_kinetics400` | Temporal Shift Module model with lightweight temporal information exchange. |
| `tsm_surrogate` | TSM Surrogate | `r50_kinetics400` | TSM-compatible surrogate model variant for transfer or diagnostic runs. |
| `tsmnonlocal` | TSM Non-Local | `r50_kinetics400` | TSM variant with non-local reasoning blocks. |
| `tsn` | TSN | `r101_kinetics400` | Temporal Segment Network baseline with segment-level aggregation. |
| `umt` | UMT | `vit_large_p16_kinetics400` | Unmasked Teacher video foundation model wrapper. |
| `uniformer` | UniFormer | `base_kinetics400` | Unified convolution-transformer video architecture. |
| `uniformerv2` | UniFormerV2 | `base_kinetics400` | Second-generation UniFormer video architecture. |
| `videomae` | VideoMAE | `vit_base_p16_kinetics400` | Masked-autoencoder-pretrained video transformer. |
| `videomaev2` | VideoMAEv2 | `vit_base_p16_kinetics400` | Improved VideoMAE-family video transformer. |
| `videoswin` | Video Swin Transformer | `base_p244_w877_kinetics400` | Swin Transformer adapted to video with shifted spatiotemporal windows. |
| `x3d` | X3D | `m_k400` | Efficient 3D convolutional network from the X3D family. |

## Attacks

Attacks operate on sampled video tensors in `[0, 255]` before model preprocessing. The full white-box track currently uses targeted variants for the ordinary benchmark folders; individual attacks can also be run untargeted through the CLI.

| CLI name | Display name | Default preset | Type | Description |
|---|---|---|---|---|
| `amifgsm` | AMI-FGSM | `default` | Gradient-based white-box | Momentum iterative FGSM variant with an adaptive momentum update. |
| `bmtc` | BMTC | `default` | Query / transfer-oriented video attack | Background Mixup-induced Temporal Consistency attack with optional offline artifact preparation. |
| `flickering` | Flickering Attack | `default` | Video-specific white-box | Optimizes temporally varying flicker perturbations with regularization. |
| `gradestv2` | GradEstV2 | `default` | Query-based black-box | NES-based black-box attack with temporal tiling, adaptive sampling, and coarse-to-fine search. |
| `ifgsm` | I-FGSM | `default` | Gradient-based white-box | Iterative Fast Gradient Sign Method under an `L_\infty` perturbation budget. |
| `korhonen_et_al` | Korhonen et al. | `default` | Gradient-based video attack | Video-oriented iterative attack following the Korhonen et al. perturbation strategy implemented in the benchmark. |
| `mifgsm` | MI-FGSM | `default` | Gradient-based white-box | Momentum Iterative FGSM, accumulating normalized gradients across steps. |
| `square` | Square Attack | `default` | Query-based black-box | Randomized square-patch black-box attack with optional guided variant. |
| `ssah` | SSAH | `default` | Perceptual / style-oriented attack | SSAH-style attack integrated through the common sampled-video attack interface. |
| `stadv` | ST-Adv | `default` | Spatiotemporal attack | Optimizes a spatial transformation field rather than only additive pixel noise. |
| `stylefool` | StyleFool | `default` | Query / style-oriented attack | Style-transfer-based attack with an offline preparation hook for expensive style optimization. |
| `uap` | UAP | `default` | Universal perturbation | Learns or applies a universal video perturbation across samples. |
| `zhang_dists` | Zhang-DISTS | `default` | Perceptual-loss white-box | Zhang-style iterative attack using a DISTS-oriented objective. |
| `zhang_lpips` | Zhang-LPIPS | `default` | Perceptual-loss white-box | Zhang-style iterative attack using an LPIPS-oriented objective. |
| `zhang_ssim` | Zhang-SSIM | `default` | Perceptual-loss white-box | Zhang-style iterative attack using an SSIM-oriented objective. |

## Defences

Defences transform sampled clips before model preprocessing. In non-adaptive evaluation, the attack is produced against the clean model and the defence is applied during evaluation. With `--adaptive`, the defence is installed before attack optimization so gradients pass through the defence whenever possible.

| CLI name | Display name | Default preset | Description |
|---|---|---|---|
| `bilateral` | Bilateral filtering | `default` | Applies spatiotemporal bilateral smoothing to suppress small perturbations while preserving edges. |
| `crop_resize` | Crop-resize | `default` | Randomly crops and resizes frames to disrupt spatially localized perturbations. |
| `diff_jpeg` | Differentiable JPEG | `default`, `strong` | Differentiable JPEG compression (soft-rounded DCT quantization); gradients flow through `transform`, making it usable as an adaptive (white-box) compression defence. |
| `domain_transform` | Domain transform filtering | `default` | Applies domain-transform-style smoothing controlled by range and temporal sigmas. |
| `flip` | Horizontal flip | `default` | Horizontally flips frames according to `flip_prob`; useful as a simple input transformation defence. |
| `freqpure` | FreqPure | `default` | Diffusion/frequency-domain purification defence with DDIM/DDPM preset options. |
| `gaussian_blur` | Gaussian blur | `default` | Applies per-frame Gaussian smoothing. |
| `jpeg_compression` | JPEG compression | `default` | Re-encodes frames through JPEG compression at a configurable quality level. |
| `randomized_smoothing` | Randomized smoothing | `default` | Applies randomized Gaussian smoothing/noise-like transformations over a configured sigma range. |
| `rotate` | Random rotation | `default` | Rotates frames within a configurable angle range. |
| `shuffle` | Temporal/spatial shuffle | `default` | Shuffles local video regions according to configured block parameters. |
| `temporal_median` | Temporal median | `default` | Applies temporal median filtering across neighboring frames. |
| `videopure` | VideoPure | `default` | Diffusion-based video purification wrapper with continuous, auto, and legacy presets. |

## Datasets

| Dataset | Subsets | Description |
|---|---|---|
| `kinetics400` | `k400_val`, `k400_test`, `kinetics400_mini_val`, `kinetics400_mini_test` | Kinetics-400 validation/test manifests and mini aliases used by quick tests and benchmark runs. |

Dataset entries are resolved from `configs/datasets.toml`. The artifact command can list and resolve them:

```bash
vcr-bench-artifacts list-dataset-subsets --dataset kinetics400
vcr-bench-artifacts resolve-dataset-subset --dataset kinetics400 --subset kinetics400_mini_val
```

## Metrics

| Metric | Meaning |
|---|---|
| Clean accuracy | Top-1 accuracy on clean videos. |
| ASR | Attack success rate, computed over clean-correct videos unless a run explicitly allows a different denominator. |
| TargetSR | Targeted success rate, counting adversarial predictions that match the target class. |
| Robust / adversarial accuracy | Remaining accuracy after the attack, computed from clean-correct minus attacked-success counts over the evaluated total. |
| MSE | Mean squared pixel error between clean and attacked videos. |
| PSNR | Peak signal-to-noise ratio; higher means smaller visible distortion. |
| SSIM | Structural similarity; higher means more similar videos. |
| LPIPS | Learned perceptual image patch similarity; lower means more similar videos. |
| DISTS | Deep Image Structure and Texture Similarity; supported in the schema, but some runs may leave it empty. |
| VMAF | Video Multi-Method Assessment Fusion; higher means better perceptual video quality. |
| Iterations | Mean number of optimization/query iterations used per video. |
| Time/video | Mean processing time for one video. |
| VRAM profile | Optional CUDA memory profiling rows for inference, gradient pass, and attack call peaks. |

Perceptual metrics are averaged over generated adversarial examples, including failed attacks. Interpret them together with ASR: a very low-distortion run can simply mean the attack failed to move the prediction.

## CLI Capabilities

| Command | Description |
|---|---|
| `vcr-bench-test` | Evaluates clean accuracy and can write JSON/CSV summaries. |
| `vcr-bench-attack` | Runs attacks, optional defences, targeted/untargeted evaluation, metrics, video dumping, and VRAM profiling. |
| `vcr-bench-classify` | Classifies arbitrary folders of videos and writes JSON/CSV predictions. |
| `vcr-bench-prepare` | Runs offline preparation hooks for attacks such as StyleFool or BMTC. |
| `vcr-bench-artifacts` | Lists, resolves, downloads, uploads, and packages checkpoints and dataset archives. |
| `vcr-bench-remote` | Synchronizes code and launches/monitors/fetches Slurm jobs on a configured remote cluster. |

Important cross-cutting features:

- Presets: `--run-preset`, `--model-preset`, `--attack-preset`, and `--defence-preset` load JSON configurations from `configs/`.
- Dotted overrides: `--override key=value` patches preset fields without copying a whole config.
- Model option discovery: `--list-model-options` prints available backbones and weight datasets.
- Attack option discovery: `--print-attack-spec` prints automatically exposed attack parameters.
- Adaptive defence evaluation: `--adaptive` installs a defence before attack optimization.
- Lite attack mode: `--lite-attack` uses a lower-cost attack pipeline when supported by the model.
- Metric controls: `--no-lpips`, `--no-vmaf`, `--framewise-metrics`, and `--metric-workers` control expensive perceptual metrics.
- Video artifacts: `--dump-freq` and `--save-defence-stages` export clean, attacked, and defended videos for inspection.
- VRAM profiling: `--vram-profile-csv` records CUDA memory peaks for reproducibility and hardware planning.
