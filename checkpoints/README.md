# Checkpoints

Place released model weights here. The default inference configuration expects
`DSC_CKD.pt` (CKD 1500) and `DSC_CS.pt` (CS 1200). Weight files are ignored by
Git.

CS no-GAN checkpoints produced by `configs/train/cs_nogan.yaml` can be used by
the same inference command. Inference loads only generator/EMA weights and
normalizers; discriminator state is optional and ignored.
