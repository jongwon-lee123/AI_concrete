# SEM-BSE DDPM dataset strategy

## Why the old crops looked arbitrary

The dataset originally used pure random crops after removing the bottom SEM overlay. That means any valid 256x256 region could be selected: aggregate-dominant regions, paste-dominant regions, pore/dark-cluster regions, or smooth regions.

For research runs that should emphasize aggregate/paste structure rather than arbitrary texture, use the structured patch sampler in `train_ddpm_bse.py`.

## Recommended first controlled experiment

1. Put only the same magnification images in the training folder.
2. Start with 50x SEM-BSE images only.
3. Keep the footer crop enabled.
4. Use structured patch sampling.

Example PowerShell command:

```powershell
& C:/ProgramData/anaconda3/envs/test/python.exe C:/Users/whddn/Desktop/AI/AI_concrete/train_ddpm_bse.py `
  --data_dir "C:\Users\whddn\Desktop\BSE_mag50" `
  --save_dir "C:\Users\whddn\Desktop\AI\AI_concrete\outputs_mag50_structure" `
  --input_selection first `
  --input_image_count 4 `
  --require_input_image_count True `
  --num_samples 9 `
  --ignore_bottom_px 180 `
  --patch_size 256 `
  --batch_size 4 `
  --patch_sampling_mode structure `
  --patch_candidate_tries 32 `
  --min_edge_fraction 0.015 `
  --min_mid_gray_fraction 0.45 `
  --min_dark_fraction 0.005 `
  --max_dark_fraction 0.35
```

## Quick 4-image / 9-sample pilot

For a fast pilot with exactly four selected 50x inputs, keep `input_image_count=4` and `num_samples=9`. The saved grid will be 3x3 instead of 4x4, which makes every sampling/evaluation pass lighter.

## Image acquisition recommendation

For the next SEM-BSE extraction campaign, keep acquisition conditions consistent before increasing model complexity.

- Use one magnification per experiment; do not mix 25x, 50x, and 65x in one unconditional DDPM run.
- Use 50x first because it captures aggregate/paste morphology while still giving enough local texture for 256x256 or 512x512 patches.
- Save raw images at the largest available pixel resolution with the same HV/current/WD/detector settings.
- Acquire at least 20 images for a controlled pilot and preferably 50 or more images for a stable generative run.
- Keep the scale bar/footer, but always crop it during training with `ignore_bottom_px`.

## How to interpret structured sampling

The structured sampler is not a physical phase segmentation model. It is a lightweight image-analysis gate that samples several random candidate patches and prefers patches with:

- enough grayscale gradients, which usually indicate aggregate/paste boundaries or textured microstructure;
- enough mid-gray pixels, so the crop is not dominated only by black pore/defect regions;
- a bounded dark-pixel fraction, so pore/dark clusters are present but not overwhelming.

This is closer to literature-style image-analysis-guided curation than pure random cropping, but phase segmentation and micromechanical validation should still be added later for publication-level evaluation.


## 3.1 phase-assemblage proxy (current code)

The training script now logs simple 3-bin phase-assemblage proxies in `eval_stats.csv` for both generated and real batches:

- `phase_dark_frac` (pixels `<= phase_dark_max`)
- `phase_mid_frac` (pixels in `(phase_dark_max, phase_mid_max]`)
- `phase_bright_frac` (pixels `> phase_mid_max`)
- `phase_entropy_3bin` (entropy of the 3-bin phase fractions)

These are grayscale proxies, not true pore/paste/aggregate segmentation labels. They are useful for checking whether generated patches drift toward too-dark or too-bright regimes during training.
