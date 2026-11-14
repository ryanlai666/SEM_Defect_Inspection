"""Shape / behaviour tests with randomly initialised weights (no downloads, CPU)."""

import torch

from semdefect.data.splits import anomaly_split  # noqa: F401  (import check)
from semdefect.kernels.knn import knn_torch
from semdefect.models import (
    AnomalyDINO,
    Dinomaly,
    DinoSegHead,
    EfficientAD,
    LinearProbe,
    PatchCore,
    PrototypeClassifier,
    UNet,
    dice_bce_loss,
)
from semdefect.models.anomaly import greedy_coreset, offmask_patches


def test_unet_variants_shapes():
    x = torch.randn(2, 3, 64, 64)
    assert UNet(base=8)(x).shape == (2, 1, 64, 64)
    assert UNet(base=8, encoder="resnet18", pretrained=False)(x).shape == (2, 1, 64, 64)


def test_dino_seg_head_freezes_encoder():
    m = DinoSegHead("dinov2_vits14", img_size=56, pretrained=False, dec_ch=16)
    y = m(torch.randn(1, 3, 56, 56))
    assert y.shape == (1, 1, 56, 56)
    assert all(not p.requires_grad for p in m.encoder.parameters())
    loss = dice_bce_loss(y, torch.zeros_like(y))
    loss.backward()
    assert m.head.weight.grad is not None


def test_knn_torch_exact():
    q, b = torch.randn(50, 16), torch.randn(300, 16)
    d, i = knn_torch(q, b, k=3, chunk=17)
    ref = torch.cdist(q, b).square()
    rv, ri = ref.topk(3, largest=False)
    assert torch.allclose(d, rv, atol=1e-4) and torch.equal(i, ri)


def test_greedy_coreset_covers_clusters():
    centers = torch.tensor([[0.0, 0], [10, 0], [0, 10], [10, 10]])
    pts = torch.cat([c + 0.1 * torch.randn(50, 2) for c in centers])
    sel = greedy_coreset(pts, 4, proj_dim=2)
    assert len({int(s) // 50 for s in sel}) == 4


def test_offmask_patches_drops_cells_near_defect():
    emb = torch.arange(16).float().view(1, 16, 1)
    mask = torch.zeros(1, 1, 32, 32)
    mask[0, 0, 0:8, 0:8] = 1  # top-left cell of a 4x4 grid
    kept = offmask_patches(emb, mask, margin_px=0)
    assert kept.shape[0] == 15 and 0 not in kept.flatten().tolist()


def _loader(n=4, size=64):
    return [
        {"image": torch.randn(2, 3, size, size), "mask": torch.zeros(2, 1, size, size)} for _ in range(n // 2)
    ]


def test_patchcore_fit_predict():
    pc = PatchCore(backbone="resnet18", layers=(2, 3), embed_dim=64, coreset_ratio=0.5, pretrained=False)
    info = pc.fit(_loader(), torch.device("cpu"), offmask_loader=_loader(2))
    assert info["memory_size"] > 0 and info["n_offmask_patches"] > 0
    s, m = pc.predict(torch.randn(2, 3, 64, 64), out_size=32)
    assert s.shape == (2,) and m.shape == (2, 32, 32)


def test_anomalydino_scores_identical_image_low():
    ad = AnomalyDINO("dinov2_vits14", img_size=56, pretrained=False, rotations=False)
    x = torch.randn(2, 3, 56, 56)
    ad.fit([{"image": x}], torch.device("cpu"))
    s_same, _ = ad.predict(x)
    s_new, maps = ad.predict(torch.randn(2, 3, 56, 56))
    assert (s_same < 1e-3).all() and (s_new > s_same).all() and maps.shape == (2, 56, 56)


def test_efficientad_losses_and_predict():
    m = EfficientAD(pretrained_teacher=False)
    x = torch.randn(1, 3, 256, 256)  # AE needs >= 256 px (5 stride-2 convs + 8x8 conv)
    losses = m.losses(x, x)
    losses["loss"].backward()
    assert m.student.conv1.weight.grad is not None and m.teacher.conv1.weight.grad is None
    m.eval()
    s, maps = m.predict(x, out_size=64)
    assert s.shape == (1,) and maps.shape == (1, 64, 64)


def test_dinomaly_loss_and_predict():
    m = Dinomaly("dinov2_vits14_reg", img_size=56, enc_layers=(2, 3, 4, 5), pretrained=False)
    x = torch.randn(2, 3, 56, 56)
    loss = m.loss(x)
    loss.backward()
    assert torch.isfinite(loss)
    m.eval()
    s, maps = m.predict(x, out_size=28)
    assert s.shape == (2,) and maps.shape == (2, 28, 28)


def test_dinomaly_offmask_loss_ignores_masked_patches():
    from semdefect.models.anomaly import offmask_keep

    torch.manual_seed(0)
    m = Dinomaly("dinov2_vits14_reg", img_size=56, enc_layers=(2, 3, 4, 5), dropout=0.0, pretrained=False)
    m.eval()  # deterministic decoder for the comparison
    x = torch.randn(2, 3, 56, 56)
    all_keep = torch.ones(2, 16, dtype=torch.bool)
    assert torch.allclose(m.loss(x), m.loss(x, keep=all_keep), atol=1e-6)
    mask = torch.zeros(2, 1, 56, 56)
    mask[0, :, 0:10, 0:10] = 1  # defect in the top-left cell of image 0
    keep = offmask_keep(mask, side=4, margin_px=0)
    assert keep.sum() == 31 and not keep[0, 0]
    # dropped patches receive exactly zero gradient; kept ones do not (stub encoder/decoder outputs)
    e = [torch.randn(2, 16, 8)]
    d = [torch.randn(2, 16, 8, requires_grad=True)]
    m.forward = lambda _x: (e, d)
    loss = m.loss(x, keep=keep)
    loss.backward()
    assert torch.isfinite(loss)
    assert d[0].grad[0, 0].abs().sum() == 0
    assert d[0].grad[keep].abs().sum() > 0


def test_probes_separate_clusters():
    rng = torch.Generator().manual_seed(0)
    x = torch.cat([torch.randn(20, 8, generator=rng) + 4 * torch.eye(8)[c] for c in range(3)]).numpy()
    y = torch.arange(3).repeat_interleave(20).numpy()
    assert (LinearProbe().fit(x, y).predict(x) == y).mean() > 0.95
    assert (PrototypeClassifier().fit(x, y).predict(x) == y).mean() > 0.95
