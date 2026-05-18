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
  --input_selection all `
  --input_image_count 0 `
  --require_input_image_count False `
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
