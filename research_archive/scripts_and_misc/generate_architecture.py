"""
Generate IEEE-quality MSDNet Architecture Diagram - v3
Portrait aspect ratio (7:9) to fill IEEE figure* page properly.
Larger text, better spacing, no cutoffs.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# -- Portrait ratio matching IEEE page (7:9) --
fig_w, fig_h = 14, 18
DPI = 300

# Colors
C_BG      = '#FFFFFF'
C_INPUT   = '#DCEEFB'
C_PREPROC = '#E0E3F1'
C_BACKBONE= '#D5EDE8'
C_FPN     = '#FFF0DB'
C_CBAM    = '#EDDDF5'
C_BRANCH  = '#D6EDDA'
C_OUTPUT  = '#FCDEDE'
C_XAI     = '#FFF5D6'
C_VESSEL  = '#DCE4E8'
C_LOSS_BG = '#FCE4EC'
C_BORDER  = '#37474F'
C_ARROW   = '#37474F'
C_TEXT    = '#1A1A1A'
C_SUB     = '#555555'
C_ACCENT  = '#1565C0'
C_LOSS    = '#B71C1C'
C_PIPE_BG = '#E3F2FD'


def box(ax, x, y, w, h, text, sub=None, color=C_INPUT, fs=10, bold=True):
    """Rounded box with centered text."""
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.02", lw=1.2,
                       ec=C_BORDER, fc=color, zorder=3)
    ax.add_patch(r)
    wt = 'bold' if bold else 'normal'
    ty = y + (0.15 if sub else 0)
    ax.text(x, ty, text, ha='center', va='center', fontsize=fs,
            fontweight=wt, color=C_TEXT, fontfamily='serif', zorder=4)
    if sub:
        ax.text(x, y - 0.15, sub, ha='center', va='center', fontsize=fs - 2,
                color=C_SUB, fontfamily='serif', style='italic', zorder=4)


def arr(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.4, style='->'):
    """Straight arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=2)


def carr(ax, x1, y1, x2, y2, rad=0.3, color=C_ARROW, lw=1.2):
    """Curved arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad={}'.format(rad)),
                zorder=2)


def heading(ax, x, y, text, fs=11):
    """Section heading with white background."""
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=C_ACCENT, fontfamily='serif', zorder=6,
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.95))


def fmap(ax, x, y, w, h, n=3, color='#B3E5FC'):
    """Stacked feature map rectangles."""
    off = 0.06
    for i in range(n - 1, -1, -1):
        a = 0.5 + 0.15 * i
        r = FancyBboxPatch((x - w/2 + i*off, y - h/2 + i*off), w, h,
                           boxstyle="round,pad=0.01", lw=0.8,
                           ec=C_BORDER, fc=color, alpha=a, zorder=2+i)
        ax.add_patch(r)


def dashed_box(ax, x, y, w, h, label='', color='#78909C'):
    """Dashed bounding box with label."""
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.03", lw=1.0,
                       ec=color, fc='none', ls='--', zorder=1)
    ax.add_patch(r)
    if label:
        ax.text(x + w/2 - 0.1, y + h/2 - 0.1, label,
                ha='right', va='top', fontsize=7.5,
                fontweight='bold', color=color, fontfamily='serif', zorder=4)


# ---- Create figure ----
fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=DPI)
ax.set_xlim(0, 14)
ax.set_ylim(0.5, 18)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

cx = 7.0  # horizontal center

# ============================================================
# TITLE  (y~17.5)
# ============================================================
ax.text(cx, 17.5, 'MSDNet: Multi-Scale Disentangled Network Architecture',
        ha='center', va='center', fontsize=15, fontweight='bold',
        color=C_TEXT, fontfamily='serif')

# ============================================================
# (1) INPUT DATASETS  (y~16.6)
# ============================================================
y1 = 16.6
heading(ax, cx, y1 + 0.4, '(1) Input Data Sources')
dsets = [
    (cx - 4.0, y1, 'APTOS 2019', '3,662 images | DR 0-4'),
    (cx,       y1, 'HRDC 2023',  '~1,200 images | HR Binary'),
    (cx + 4.0, y1, 'IDRiD + DRIVE', '556 images | DR + Vessels'),
]
for dx, dy, t, s in dsets:
    box(ax, dx, dy, 3.2, 0.55, t, s, C_INPUT, fs=9)

# arrows merging down
for dx, _, _, _ in dsets:
    arr(ax, dx, y1 - 0.28, cx, y1 - 0.75)

# ============================================================
# (2) PREPROCESSING  (y~15.4)
# ============================================================
y2 = 15.35
heading(ax, cx, y2 + 0.35, '(2) Preprocessing Pipeline')
preps = [
    (cx - 4.0, y2, 'Ben Graham\nNormalization'),
    (cx - 1.35, y2, 'CLAHE\nEnhancement'),
    (cx + 1.35, y2, 'Optic Disc\nSuppression'),
    (cx + 4.0, y2, 'Resize 224x224\n+ Augmentation'),
]
for px, py, t in preps:
    box(ax, px, py, 2.4, 0.48, t, color=C_PREPROC, fs=8, bold=False)
# chain arrows
for i in range(len(preps) - 1):
    arr(ax, preps[i][0] + 1.2, preps[i][1], preps[i+1][0] - 1.2, preps[i+1][1], lw=1.0)
dashed_box(ax, cx, y2, 11.0, 0.8, 'Preprocessing', '#7986CB')

# arrow down
arr(ax, cx, y2 - 0.4, cx, y2 - 0.9)

# ============================================================
# (3) BACKBONE  (y~14.0)
# ============================================================
y3 = 14.0
heading(ax, cx, y3 + 0.35, '(3) Feature Extraction Backbone')
box(ax, cx, y3, 6.0, 0.5, 'EfficientNet-B0  (ImageNet Pretrained)', color=C_BACKBONE, fs=11)

# three arrows to features
arr(ax, cx, y3 - 0.25, cx - 3.5, y3 - 0.9)
arr(ax, cx, y3 - 0.25, cx, y3 - 0.9)
arr(ax, cx, y3 - 0.25, cx + 3.5, y3 - 0.9)

# ============================================================
# (4) MULTI-SCALE FEATURES  (y~12.7)
# ============================================================
y4 = 12.7
heading(ax, cx, y4 + 0.4, '(4) Multi-Scale Feature Maps')
feats = [
    (cx - 3.5, y4, 'P3: 40 x 28 x 28', 'Stride 8',  '#B3E5FC'),
    (cx,       y4, 'P4: 112 x 14 x 14','Stride 16', '#81D4FA'),
    (cx + 3.5, y4, 'P5: 320 x 7 x 7',  'Stride 32', '#4FC3F7'),
]
for fx, fy, t, s, c in feats:
    fmap(ax, fx, fy, 2.4, 0.45, 3, c)
    ax.text(fx + 0.08, fy + 0.08, t, ha='center', va='center',
            fontsize=8, fontweight='bold', color=C_TEXT, fontfamily='serif', zorder=10)
    ax.text(fx + 0.08, fy - 0.14, s, ha='center', va='center',
            fontsize=7, color=C_SUB, fontfamily='serif', zorder=10)

# arrows to FPN
for fx, _, _, _, _ in feats:
    arr(ax, fx, y4 - 0.25, cx, y4 - 0.8)

# ============================================================
# (5) FPN  (y~11.5)
# ============================================================
y5 = 11.5
heading(ax, cx, y5 + 0.35, '(5) Feature Pyramid Network (FPN)')
box(ax, cx, y5, 7.0, 0.5, 'FPN: 1x1 Conv + Top-Down Pathway + 3x3 Smooth',
    'Output: F_fpn  [B x 256 x 28 x 28]', C_FPN, fs=9.5)

# split point
arr(ax, cx, y5 - 0.25, cx, y5 - 0.7)
ax.plot(cx, y5 - 0.7, 'o', color=C_BORDER, ms=5, zorder=5)

# ============================================================
# (6) DISEASE BRANCHES  (y~9.5)
# ============================================================
y6 = 9.7
heading(ax, cx, y6 + 0.7, '(6) Disease-Specific Classification Branches')
ax.text(cx, y6 + 0.4, 'Each branch: CBAM -> GAP -> MC Dropout -> Classifier',
        ha='center', va='center', fontsize=8, color=C_SUB,
        fontfamily='serif', style='italic', zorder=6,
        bbox=dict(boxstyle='round,pad=0.06', fc='white', ec='none', alpha=0.9))

# --- DR Branch (left) ---
dr_x = cx - 3.2
arr(ax, cx, y5 - 0.7, dr_x, y6 + 0.15)

box(ax, dr_x, y6, 3.2, 0.42, 'CBAM_DR', 'Channel + Spatial Attention', C_CBAM, fs=9)
arr(ax, dr_x, y6 - 0.21, dr_x, y6 - 0.5)
box(ax, dr_x, y6 - 0.72, 3.2, 0.38, 'GAP -> MC Dropout (p=0.3)', color=C_BRANCH, fs=8, bold=False)
arr(ax, dr_x, y6 - 0.91, dr_x, y6 - 1.2)
box(ax, dr_x, y6 - 1.42, 3.2, 0.38, 'DR Classifier (5-class)', 'Focal Loss', C_BRANCH, fs=9)

# --- HR Branch (right) ---
hr_x = cx + 3.2
arr(ax, cx, y5 - 0.7, hr_x, y6 + 0.15)

box(ax, hr_x, y6, 3.2, 0.42, 'CBAM_HR', 'Channel + Spatial Attention', C_CBAM, fs=9)
arr(ax, hr_x, y6 - 0.21, hr_x, y6 - 0.5)
box(ax, hr_x, y6 - 0.72, 3.2, 0.38, 'GAP -> MC Dropout (p=0.3)', color=C_BRANCH, fs=8, bold=False)
arr(ax, hr_x, y6 - 0.91, hr_x, y6 - 1.2)
box(ax, hr_x, y6 - 1.42, 3.2, 0.38, 'HR Classifier (Binary)', 'BCE Loss', C_BRANCH, fs=9)

# Contrastive loss between branches
y_dis = y6 - 0.72
carr(ax, dr_x + 1.6, y_dis, hr_x - 1.6, y_dis, rad=-0.12, color=C_LOSS, lw=1.3)
carr(ax, hr_x - 1.6, y_dis, dr_x + 1.6, y_dis, rad=-0.12, color=C_LOSS, lw=1.3)
ax.text(cx, y_dis + 0.22, 'L_dis (Contrastive\nDisentanglement)',
        ha='center', va='center', fontsize=7.5, fontweight='bold',
        color=C_LOSS, fontfamily='serif', zorder=5,
        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec=C_LOSS, alpha=0.95, lw=1.0))

dashed_box(ax, cx, y6 - 0.6, 9.5, 2.3, 'Disease Branches', '#7E57C2')

# --- Vessel Decoder (right side, inside bounds) ---
vx = cx + 7.8  # was overflowing - pull in
vy = y6 - 0.3
# arrow from FPN split
carr(ax, cx + 3.5, y5, cx + 6.5, vy + 0.25, rad=-0.2, color='#607D8B', lw=1.0)
# Pulled left to x=13.2 max to stay in bounds (xlim=14)
box(ax, 12.0, vy, 2.6, 0.55, 'Vessel Decoder', 'U-Net (Train Only)', C_VESSEL, fs=8)
arr(ax, 12.0, vy - 0.28, 12.0, vy - 0.65, color='#607D8B', lw=1.0)
box(ax, 12.0, vy - 0.85, 2.6, 0.35, 'Dice + BCE Loss', 'DRIVE Masks', C_VESSEL, fs=7, bold=False)

# ============================================================
# (7) PREDICTIONS  (y~7.0)
# ============================================================
y7 = 7.0
heading(ax, cx, y7 + 0.65, '(7) Predictions + Uncertainty Estimation')

arr(ax, dr_x, y6 - 1.61, dr_x, y7 + 0.35)
arr(ax, hr_x, y6 - 1.61, hr_x, y7 + 0.35)

# DR grade boxes
grades = ['Grade 0\n(No DR)', 'Grade 1\n(Mild)', 'Grade 2\n(Moderate)',
          'Grade 3\n(Severe)', 'Grade 4\n(PDR)']
g_start = dr_x - 2.6
for i, g in enumerate(grades):
    gx = g_start + i * 1.35
    box(ax, gx, y7, 1.2, 0.5, g, color=C_OUTPUT, fs=6.5, bold=False)

# HR output
box(ax, hr_x, y7, 3.0, 0.5, 'HR: Present / Absent', color=C_OUTPUT, fs=9)

# Uncertainty
box(ax, hr_x + 3.3, y7, 2.0, 0.5, 'Uncertainty\n(sigma)', color='#ECEFF1', fs=7.5, bold=False)
arr(ax, hr_x + 1.5, y7, hr_x + 2.3, y7, lw=1.0)
ax.text(hr_x + 3.3, y7 - 0.38, 'N = 30 MC passes', ha='center', va='center',
        fontsize=6.5, color=C_SUB, fontfamily='serif', style='italic')

# ============================================================
# (8) EXPLAINABILITY  (y~5.3)
# ============================================================
y8 = 5.3
heading(ax, cx, y8 + 0.4, '(8) Per-Branch Explainability')
box(ax, cx, y8, 5.5, 0.5, 'Per-Branch Grad-CAM++',
    'Separate Heatmaps for DR & HR', C_XAI, fs=10)

arr(ax, dr_x, y7 - 0.25, cx - 1.5, y8 + 0.25, lw=1.0)
arr(ax, hr_x, y7 - 0.25, cx + 1.5, y8 + 0.25, lw=1.0)

# heatmap outputs
y_hm = y8 - 0.7
box(ax, cx - 3.0, y_hm, 3.0, 0.42, 'DR Heatmap', 'Hemorrhages, Exudates', '#FFF9C4', fs=8)
box(ax, cx + 3.0, y_hm, 3.0, 0.42, 'HR Heatmap', 'Vessel Crossings, Disc', '#FFF9C4', fs=8)
arr(ax, cx - 1.0, y8 - 0.25, cx - 3.0, y_hm + 0.21, lw=1.0)
arr(ax, cx + 1.0, y8 - 0.25, cx + 3.0, y_hm + 0.21, lw=1.0)

# ============================================================
# (9) CLINICAL PIPELINE  (y~3.3)
# ============================================================
y9 = 3.3
heading(ax, cx, y9 + 0.45, '(9) IRDAS Clinical Deployment Pipeline')

stages = [
    (cx - 4.0, y9, 'Stage 1: Triage', 'XGBoost + SHAP', '#E3F2FD'),
    (cx,       y9, 'Stage 2: MSDNet', 'Multi-Task Analysis', C_BACKBONE),
    (cx + 4.0, y9, 'Stage 3: Reports', 'Gemini Pro / 9 Languages', '#FFF3E0'),
]
for sx, sy, t, s, c in stages:
    box(ax, sx, sy, 3.2, 0.55, t, s, c, fs=9)
arr(ax, stages[0][0] + 1.6, y9, stages[1][0] - 1.6, y9)
arr(ax, stages[1][0] + 1.6, y9, stages[2][0] - 1.6, y9)
dashed_box(ax, cx, y9, 12.0, 0.9, 'End-to-End Pipeline', '#43A047')

# ============================================================
# (10) TOTAL LOSS  (y~2.0)
# ============================================================
y10 = 2.0
box(ax, cx, y10, 11.5, 0.45,
    'L_total = L_DR (Focal) + L_HR (BCE) + 0.5 * L_vessel (Dice+BCE) + 0.3 * L_dis (Contrastive)',
    color=C_LOSS_BG, fs=9)
ax.text(cx, y10 - 0.4,
        'Optimizer: AdamW  |  lr = 1e-4  |  Scheduler: Cosine Annealing  |  50 Epochs  |  Batch Size: 32',
        ha='center', va='center', fontsize=7.5, color=C_SUB, fontfamily='serif')

# ---- Save ----
plt.tight_layout(pad=0.3)
out = r'c:\Users\shive\projects\cv\retinal-disease-detection-cv\figures\fig1_architecture'
fig.savefig(out + '.png', dpi=DPI, bbox_inches='tight', facecolor=C_BG, pad_inches=0.2)
fig.savefig(out + '.pdf', format='pdf', bbox_inches='tight', facecolor=C_BG, pad_inches=0.2)
print("Saved: " + out + ".png")
print("Saved: " + out + ".pdf")
print("Dimensions: {} x {} px @ {} DPI".format(fig_w * DPI, fig_h * DPI, DPI))
plt.close()
