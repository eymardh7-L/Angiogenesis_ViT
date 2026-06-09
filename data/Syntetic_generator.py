#!/usr/bin/env python3
"""
CAM Synthetic Dataset Generator
===============================================================
Generates histopathology-style synthetic images of the chick chorioallantoic
membrane (CAM) with a tumor-induced vascular network. Vessels are rendered in
an intense red-maroon tone for clearer biological visualization.

The generator is fully self-contained: it requires no external image files and
no real biological or clinical data. Every image is produced algorithmically
from a random seed, so the entire dataset is bit-for-bit reproducible.

Usage: paste this whole script into a Jupyter cell and run it, or import the
`CAMGenerator` class and the `generate_v1` / `generate_v2` helpers.

Dependencies: numpy, scipy, Pillow, matplotlib
"""

import numpy as np
from scipy.ndimage import gaussian_filter, rotate, zoom
from PIL import Image
import os
import json
import matplotlib.pyplot as plt

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def bezier_noise(p0, p1, p2, n=50, noise_amp=2.0):
    """Quadratic Bezier curve perturbed by spatially correlated noise.

    The Bezier curve gives a smooth backbone between the three control points,
    while the Gaussian-smoothed noise added on top mimics the irregular,
    biologically realistic meandering of a real blood vessel (rather than a
    perfectly geometric arc).

    Args:
        p0, p1, p2: (x, y) start, control, and end points of the curve.
        n:          number of sample points along the curve.
        noise_amp:  standard deviation of the random displacement before
                    smoothing; larger values produce wigglier vessels.

    Returns:
        (x, y) arrays with the noisy curve coordinates.
    """
    t = np.linspace(0, 1, n)
    # Standard quadratic Bezier parameterization.
    x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
    y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
    # Correlated perturbation: white noise smoothed by a Gaussian kernel so the
    # displacement varies gradually along the vessel instead of pixel-to-pixel.
    nx = gaussian_filter(np.random.normal(0, noise_amp, n), sigma=2)
    ny = gaussian_filter(np.random.normal(0, noise_amp, n), sigma=2)
    return x + nx, y + ny


# ============================================================
# GENERATOR CLASS
# ============================================================

class CAMGenerator:
    """Synthetic image generator for the chorioallantoic membrane (CAM).

    Two biological knobs drive the appearance of each image:
        alpha: angiogenic intensity   (0-1) -- higher means denser vasculature.
        beta:  anti-vascular intensity (0-1) -- higher means stronger drug
               inhibition (sparser, thinner, more tortuous vessels).
    """

    def __init__(self, R=140):
        # R is the working radius (in pixels) of the circular CAM field; the
        # internal canvas is (2R+1) x (2R+1) and is downscaled at render time.
        self.R = R
        # Center of the circular field.
        self.C = np.array([R, R], dtype=np.float32)

    def vessel(self, angle, alpha=1.0, beta=0.0, gen=0, tort=1.0, cal=1.0):
        """Generate a single vessel (with possible child branches).

        Args:
            angle: entry angle (radians) at which the vessel enters the field.
            alpha: angiogenic intensity (controls branching probability).
            beta:  anti-vascular intensity (thins vessels, increases tortuosity).
            gen:   branching generation (0 = root vessel, 1 = first branch, ...).
            tort:  global tortuosity multiplier.
            cal:   global caliber (thickness) multiplier.

        Returns:
            (x, y, thick, branches) where x/y are the vessel path, thick is the
            per-point line width, and branches is a list of seeds for child
            vessels: (start_x, start_y, branch_angle, next_generation).
        """
        R, C = self.R, self.C

        # Start point near the outer rim of the field.
        r0 = R * (0.84 + 0.12 * np.random.rand())
        p0 = C + r0 * np.array([np.cos(angle), np.sin(angle)])

        # End point near the central tumor; drug inhibition (beta) pushes the
        # endpoint slightly outward, i.e. vessels fail to reach the tumor.
        r2 = R * (0.08 + 0.14 * np.random.rand() + 0.10 * beta)
        ae = angle + np.pi + np.random.normal(0, 0.6 * (1 + gen * 0.3))
        p2 = C + r2 * np.array([np.cos(ae), np.sin(ae)])

        # Control point that bends the vessel; bending grows with beta.
        mid = (p0 + p2) / 2
        perp = np.array([-np.sin(angle), np.cos(angle)])
        p1 = mid + perp * R * np.random.normal(0, 0.15 * (1 + 0.4 * beta))

        # Biological noise amplitude: more tortuous under inhibition (beta).
        na = (1.5 + 2.5 * beta) * tort
        # Number of points along the vessel; fewer when heavily inhibited and
        # for deeper branch generations.
        npts = 40 + int(35 * (1 - beta * 0.8) * (0.9 ** gen))
        x, y = bezier_noise(p0, p1, p2, n=npts, noise_amp=na)

        # Vessel caliber tapers from thick (proximal) to thin (distal); base
        # thickness shrinks with inhibition (beta) and with branch generation.
        bt = (3.0 - 1.0 * beta) * (0.80 ** gen) * cal
        thick = bt * np.exp(-2.2 * np.linspace(0, 1, npts)) + 0.5

        # Branching: only the first two generations may spawn children. The
        # branch count is Poisson-distributed and scales with alpha (more
        # angiogenesis) and shrinks with beta (drug suppresses branching).
        branches = []
        if gen < 2:
            nb = min(np.random.poisson(0.6 * alpha * (1 - 0.5 * beta) * (0.85 ** gen)), 2)
            for _ in range(nb):
                # Choose a branch point along the parent vessel.
                idx = np.random.randint(5, npts - 8)
                # Branch direction roughly continues outward, with a random
                # offset of about +/- pi/3.5 (a biologically typical angle).
                bd = np.arctan2(y[idx] - C[1], x[idx] - C[0]) + np.pi
                ba = bd + np.random.choice([-1, 1]) * (np.pi / 3.5 + np.random.uniform(-0.4, 0.4))
                branches.append((x[idx], y[idx], ba, gen + 1))

        return x, y, thick, branches

    def network(self, alpha=1.0, beta=0.0, seed=None, tort=1.0, cal=1.0, n_roots=None):
        """Generate the complete vascular network via breadth-first growth.

        Root vessels are seeded around the rim, then each vessel's branches are
        expanded in FIFO (queue) order until no more branches remain.

        Args:
            alpha, beta, tort, cal: see `vessel`.
            seed:    RNG seed for reproducibility.
            n_roots: number of root vessels; if None, derived from alpha/beta.

        Returns:
            A list of segments (x, y, thick, generation).
        """
        if seed is not None:
            np.random.seed(seed)
        R = self.R

        # Root count grows with angiogenesis (alpha) and is suppressed by the
        # drug (beta); clamped to a sensible range.
        if n_roots is None:
            n_roots = int(3 + 2.5 * alpha * (1 - 0.5 * beta))
        n_roots = max(2, min(n_roots, 8))

        # Distribute root entry angles around the circle and enforce a minimum
        # angular separation so roots are not bunched together.
        angs = np.sort(np.random.uniform(0, 2 * np.pi, n_roots))
        for i in range(1, len(angs)):
            if angs[i] - angs[i-1] < 0.35:
                angs[i] = angs[i-1] + 0.35 + np.random.uniform(0, 0.3)

        # Breadth-first expansion of the vessel tree.
        segs, queue = [], [(angs[i], 0) for i in range(n_roots)]
        while queue:
            a, g = queue.pop(0)
            x, y, t, br = self.vessel(a, alpha, beta, g, tort, cal)
            # Discard degenerate (too-short) segments.
            if len(x) > 5:
                segs.append((x, y, t, g))
            # Enqueue child branches for later expansion.
            for bx, by, ba, bg in br:
                queue.append((ba, bg))
        return segs

    def render(self, segs, alpha=1.0, beta=0.0, seed=None,
               bg='standard', illum=1.0, contrast=1.0, size=256):
        """Render a vascular network into an RGB image array.

        Args:
            segs:     list of segments from `network`.
            alpha, beta: biological intensities (beta darkens/desaturates vessels).
            seed:     RNG seed for reproducible background texture.
            bg:       background preset: 'standard', 'dark', 'yellowish', 'pale'.
            illum:    brightness factor (~0.75-1.25) simulating exposure.
            contrast: contrast factor (~0.75-1.35).
            size:     output side length in pixels.

        Returns:
            A float32 RGB array in [0, 1] of shape (size, size, 3).
        """
        if seed is not None:
            np.random.seed(seed)
        R = self.R

        # Blank canvas (white) at internal working resolution.
        canvas = np.ones((2 * R + 1, 2 * R + 1, 3), dtype=np.float32)

        # Predefined backgrounds: each entry is (membrane_color, tumor_color)
        # as RGB triples, emulating different tissue staining / lighting.
        presets = {
            'standard': ([0.92, 0.89, 0.82], [0.96, 0.94, 0.92]),
            'dark': ([0.85, 0.82, 0.75], [0.90, 0.88, 0.86]),
            'yellowish': ([0.95, 0.90, 0.78], [0.98, 0.95, 0.88]),
            'pale': ([0.96, 0.94, 0.90], [0.98, 0.97, 0.95]),
        }
        base, tumor = presets.get(bg, presets['standard'])

        # Two-scale Gaussian noise gives the membrane a mottled tissue texture:
        # n1 (fine grain) + n2 (coarse, low-frequency variation).
        n1 = gaussian_filter(np.random.normal(0, 0.035, (2 * R + 1, 2 * R + 1)), sigma=2.5)
        n2 = gaussian_filter(np.random.normal(0, 0.028, (2 * R + 1, 2 * R + 1)), sigma=6)
        for ch in range(3):
            canvas[:, :, ch] = base[ch] + n1 + 0.5 * n2

        # Central tumor: a filled disc (tm) plus a slightly darker rim (te);
        # everything outside the circular field is set to pure white.
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        dist = np.sqrt(xx ** 2 + yy ** 2)
        tr = 0.12 * R * np.random.uniform(0.85, 1.15)   # tumor radius
        tm, te = dist < tr, (dist >= tr) & (dist < 1.4 * tr)
        for ch in range(3):
            canvas[tm, ch] = tumor[ch]
            canvas[te, ch] = tumor[ch] - 0.03
        canvas[dist > R] = 1.0

        # Apply photographic contrast (around mid-gray 0.5) then brightness.
        canvas = np.clip((canvas - 0.5) * contrast + 0.5, 0, 1) * illum
        canvas = np.clip(canvas, 0, 1)

        # Draw vessels in intense red. Base redness decreases with inhibition.
        rb = 0.78 - 0.22 * beta
        # Sort by generation so root vessels are drawn first (children on top).
        for x, y, thick, gen in sorted(segs, key=lambda s: s[3]):
            for i in range(len(x) - 1):
                x0, y0, x1, y1 = int(x[i]), int(y[i]), int(x[i + 1]), int(y[i + 1])
                # Local line width from the average caliber of the two endpoints.
                lw = max(1, int((thick[i] + thick[i + 1]) / 2))
                # Red intensity fades slightly with generation and along length.
                r_intensity = min(1.0, rb * (1 - 0.08 * gen) * (0.9 + 0.1 * (1 - i / len(x))))

                # Rasterize the segment by sampling points between the endpoints.
                length = max(abs(x1 - x0), abs(y1 - y0)) + 1
                if length < 2:
                    continue
                t = np.linspace(0, 1, length)
                xs, ys = (x0 + t * (x1 - x0)).astype(int), (y0 + t * (y1 - y0)).astype(int)

                # Stamp a filled disc of radius lw at each sampled point to give
                # the vessel a rounded cross-section.
                for j in range(len(xs)):
                    for dx in range(-lw, lw + 1):
                        for dy in range(-lw, lw + 1):
                            if dx * dx + dy * dy <= lw * lw + 0.5:
                                px, py = xs[j] + dx, ys[j] + dy
                                if 0 <= px < 2 * R + 1 and 0 <= py < 2 * R + 1:
                                    # Paint vessel color (intense red): boost the
                                    # red channel, suppress green/blue. Slight
                                    # green/blue lift with beta desaturates the
                                    # vessel under drug inhibition.
                                    canvas[py, px, 0] = max(canvas[py, px, 0], r_intensity * 0.95)
                                    canvas[py, px, 1] = min(canvas[py, px, 1], 0.08 + 0.06 * beta)
                                    canvas[py, px, 2] = min(canvas[py, px, 2], 0.06 + 0.04 * beta)

        canvas = np.clip(canvas, 0, 1)
        # Downscale from the (2R+1) working canvas to the requested output size.
        factor = size / (2 * R + 1)
        return np.clip(zoom(canvas, (factor, factor, 1), order=1), 0, 1)


# ============================================================
# DATASET GENERATION FUNCTIONS
# ============================================================

def generate_v1(output_dir='./cam_dataset_v1', n_per_class=100, img_size=256):
    """Generate the balanced V1 dataset.

    Produces 4 classes x n_per_class images, balanced for use in:
      - Zero-shot learning
      - Few-shot learning (1, 5, 10, 20-shot)
      - Fine-tuning

    Each image gets a deterministic seed, so the dataset is fully reproducible.
    """
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    gen = CAMGenerator(R=140)

    # Class definitions: increasing anti-vascular intensity (beta) from the
    # untreated control through low- and high-dose drug conditions.
    classes = {
        'control': {'alpha': 1.0, 'beta': 0.0, 'label': 0},
        'fak_inhibition': {'alpha': 1.0, 'beta': 0.28, 'label': 1},
        'src_low': {'alpha': 1.0, 'beta': 0.55, 'label': 2},
        'src_high': {'alpha': 1.0, 'beta': 0.88, 'label': 3},
    }

    metadata = []
    for class_name, params in classes.items():
        print(f"Generating '{class_name}' ({n_per_class} images)...")
        for i in range(n_per_class):
            # Deterministic per-image seed -> exact reproducibility.
            seed = (list(classes.keys()).index(class_name) * n_per_class + i) * 12345 + 42
            np.random.seed(seed)

            # Biological variation around the class's nominal parameters.
            alpha = params['alpha'] * np.random.uniform(0.85, 1.15)
            beta = np.clip(params['beta'] * np.random.uniform(0.9, 1.1), 0, 1)
            tort = np.random.uniform(0.7, 1.4)
            cal = np.random.uniform(0.8, 1.3)
            n_roots = np.random.randint(3, 7)

            # Photographic / acquisition variation.
            bg = np.random.choice(['standard', 'dark', 'yellowish', 'pale'])
            illum = np.random.uniform(0.85, 1.15)
            contrast = np.random.uniform(0.85, 1.25)

            segs = gen.network(alpha=alpha, beta=beta, seed=seed, tort=tort, cal=cal, n_roots=n_roots)
            img = gen.render(segs, alpha=alpha, beta=beta, seed=seed, bg=bg, illum=illum, contrast=contrast, size=img_size)

            fname = f"{class_name}_{i:04d}.png"
            Image.fromarray((img * 255).astype(np.uint8)).save(f"{output_dir}/images/{fname}")

            # Record every parameter so each image can be regenerated exactly.
            metadata.append({
                'filename': fname, 'class_name': class_name, 'label': params['label'],
                'alpha': float(alpha), 'beta': float(beta), 'tortuosity': float(tort),
                'caliber': float(cal), 'n_roots': int(n_roots), 'n_segments': len(segs),
                'bg_type': bg, 'illumination': float(illum), 'contrast': float(contrast),
                'seed': seed, 'version': 'v1_balanced'
            })

    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    _create_splits_v1(metadata, output_dir)
    print(f"\nDataset V1: {len(metadata)} images in {output_dir}")
    return metadata


def generate_v2(output_dir='./cam_dataset_v2', n_total=10000, img_size=256):
    """Generate the large-scale V2 dataset for intensive training.

    Produces 4 classes x (n_total / 4) images, with the anti-vascular
    intensity (beta) sampled from a continuous range per class and an extra
    random rotation for augmentation.
    """
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    gen = CAMGenerator(R=140)

    # Per-class sampling ranges for alpha and beta (continuous, non-overlapping
    # beta bands separate the four classes).
    configs = [
        {'name': 'control', 'alpha_r': (0.8, 1.2), 'beta_r': (0.0, 0.08), 'label': 0},
        {'name': 'fak_inhibition', 'alpha_r': (0.8, 1.2), 'beta_r': (0.15, 0.40), 'label': 1},
        {'name': 'src_low', 'alpha_r': (0.8, 1.2), 'beta_r': (0.40, 0.70), 'label': 2},
        {'name': 'src_high', 'alpha_r': (0.8, 1.2), 'beta_r': (0.70, 0.98), 'label': 3},
    ]

    n_per = n_total // 4
    metadata = []

    for config in configs:
        print(f"Generating '{config['name']}' ({n_per} images)...")
        for i in range(n_per):
            # Deterministic seed (different multiplier/offset from V1).
            seed = (config['label'] * n_per + i) * 99991 + 777
            np.random.seed(seed)

            # Wider biological and photographic variation than V1.
            alpha = np.random.uniform(*config['alpha_r'])
            beta = np.random.uniform(*config['beta_r'])
            tort = np.random.uniform(0.5, 1.8)
            cal = np.random.uniform(0.6, 1.5)
            n_roots = np.random.randint(2, 8)
            bg = np.random.choice(['standard', 'dark', 'yellowish', 'pale'])
            illum = np.random.uniform(0.75, 1.25)
            contrast = np.random.uniform(0.75, 1.35)
            rot = np.random.uniform(-15, 15)   # rotation augmentation (degrees)

            segs = gen.network(alpha=alpha, beta=beta, seed=seed, tort=tort, cal=cal, n_roots=n_roots)
            img = gen.render(segs, alpha=alpha, beta=beta, seed=seed, bg=bg, illum=illum, contrast=contrast, size=img_size)

            # Apply rotation augmentation (skip negligible rotations).
            if abs(rot) > 1:
                img = rotate(img, rot, reshape=False, order=1)
                img = np.clip(img, 0, 1)

            fname = f"{config['name']}_{i:05d}.png"
            Image.fromarray((img * 255).astype(np.uint8)).save(f"{output_dir}/images/{fname}")

            metadata.append({
                'filename': fname, 'class_name': config['name'], 'label': config['label'],
                'alpha': float(alpha), 'beta': float(beta), 'tortuosity': float(tort),
                'caliber': float(cal), 'n_roots': int(n_roots), 'n_segments': len(segs),
                'bg_type': bg, 'illumination': float(illum), 'contrast': float(contrast),
                'rotation': float(rot), 'seed': seed, 'version': 'v2_massive'
            })

            # Periodic progress report for the long V2 run.
            if (i + 1) % 500 == 0:
                print(f"  ... {i + 1}/{n_per} completed")

    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    _create_splits_v2(metadata, output_dir)
    print(f"\nDataset V2: {len(metadata)} images in {output_dir}")
    return metadata


def _create_splits_v1(metadata, output_dir):
    """Build the zero-shot, few-shot, and fine-tuning splits for V1."""
    np.random.seed(42)
    # Group filenames by class.
    by_class = {}
    for m in metadata:
        by_class.setdefault(m['class_name'], []).append(m['filename'])

    # Randomly designate 2 "seen" and 2 "unseen" classes for zero-shot.
    all_classes = list(by_class.keys())
    np.random.shuffle(all_classes)
    seen, unseen = all_classes[:2], all_classes[2:]

    # Zero-shot: train/val/test on seen classes; held-out unseen classes for
    # generalization testing.
    zero_shot = {
        'seen_classes': seen, 'unseen_classes': unseen,
        'train': [f for c in seen for f in by_class[c][:70]],
        'val': [f for c in seen for f in by_class[c][70:85]],
        'test_seen': [f for c in seen for f in by_class[c][85:]],
        'test_unseen': [f for c in unseen for f in by_class[c]]
    }

    # Few-shot: k support examples per class, the rest used as the query set.
    few_shot = {}
    for k in [1, 5, 10, 20]:
        support, query = [], []
        for c, files in by_class.items():
            np.random.shuffle(files)
            support.extend(files[:k])
            query.extend(files[k:])
        few_shot[f'{k}_shot'] = {'support': support, 'query': query}

    # Fine-tuning: standard 80/10/10 train/val/test split, stratified by class.
    train, val, test = [], [], []
    for c, files in by_class.items():
        np.random.shuffle(files)
        n = len(files)
        train.extend(files[:int(0.8*n)])
        val.extend(files[int(0.8*n):int(0.9*n)])
        test.extend(files[int(0.9*n):])

    splits = {
        'zero_shot': zero_shot,
        'few_shot': few_shot,
        'fine_tuning': {'train': train, 'val': val, 'test': test}
    }

    with open(f"{output_dir}/splits.json", 'w') as f:
        json.dump(splits, f, indent=2)
    print("V1 splits saved: zero-shot, few-shot (1/5/10/20), fine-tuning")


def _create_splits_v2(metadata, output_dir):
    """Build stratified train/val/test splits for the large V2 dataset."""
    np.random.seed(42)
    # Group filenames by class.
    by_class = {}
    for m in metadata:
        by_class.setdefault(m['class_name'], []).append(m['filename'])

    # Stratified 70/15/15 split per class.
    train, val, test = [], [], []
    for c, files in by_class.items():
        np.random.shuffle(files)
        n = len(files)
        train.extend(files[:int(0.7*n)])
        val.extend(files[int(0.7*n):int(0.85*n)])
        test.extend(files[int(0.85*n):])

    splits = {
        'train': train, 'val': val, 'test': test,
        'stats': {c: len(f) for c, f in by_class.items()}
    }

    with open(f"{output_dir}/splits.json", 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"V2 splits saved: train={len(train)}, val={len(val)}, test={len(test)}")


# ============================================================
# USAGE EXAMPLES
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("CAM SYNTHETIC DATASET GENERATOR")
    print("=" * 60)
    print("\nUsage examples:")
    print("  # Generate a single test image")
    print("  gen = CAMGenerator(R=140)")
    print("  segs = gen.network(alpha=1.0, beta=0.0, seed=42)")
    print("  img = gen.render(segs, alpha=1.0, beta=0.0, seed=42)")
    print("  Image.fromarray((img * 255).astype(np.uint8)).save('test.png')")
    print("\n  # Generate the V1 dataset (400 images)")
    print("  meta = generate_v1('./cam_v1', n_per_class=100)")
    print("\n  # Generate the V2 dataset (10000 images)")
    print("  meta = generate_v2('./cam_v2', n_total=10000)")