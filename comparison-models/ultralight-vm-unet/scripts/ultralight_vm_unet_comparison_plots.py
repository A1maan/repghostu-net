import matplotlib.pyplot as plt
import numpy as np

# UltraLight-VM-UNet Results from ISIC2017 and ISIC2018 evaluations
# Placeholder values - to be filled after training and evaluation
isic2017_results = {
    'mIoU': 81.36,
    'DSC': 88.66,
    'Sensitivity': 90.51,
    'Specificity': 98.00
}

isic2018_results = {
    'mIoU': 80.24,
    'DSC': 87.83,
    'Sensitivity': 91.03,
    'Specificity': 96.27
}

# Create comparison plot
metrics = list(isic2017_results.keys())
isic2017_values = list(isic2017_results.values())
isic2018_values = list(isic2018_results.values())

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width/2, isic2017_values, width, label='ISIC2017', color='#2E86C1', alpha=0.8)
bars2 = ax.bar(x + width/2, isic2018_values, width, label='ISIC2018', color='#E74C3C', alpha=0.8)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax.set_title('UltraLight-VM-UNet Performance Comparison: ISIC2017 vs ISIC2018', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 105)

# Add subtle background color
ax.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('ultralight_vm_unet_isic_comparison_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
print("📊 Comparison plot saved as 'ultralight_vm_unet_isic_comparison_metrics.png'")
plt.show()

# Print summary comparison
print("\n" + "="*60)
print("ULTRALIGHT-VM-UNET PERFORMANCE COMPARISON")
print("="*60)
print("Metric          ISIC2017    ISIC2018    Difference")
print("-" * 60)
for metric in metrics:
    diff = isic2018_results[metric] - isic2017_results[metric]
    sign = "+" if diff > 0 else ""
    print(f"{metric:<15} {isic2017_results[metric]:>7.2f}%   {isic2018_results[metric]:>7.2f}%   {sign}{diff:>6.2f}%")
print("="*60)

# Additional analysis
print("\n📈 ANALYSIS:")
print("• UltraLight-VM-UNet with 6-stage U-shape and quad-parallel Mamba layers")
print("• Architecture: SAB + CAB skip bridges, channels [8,16,24,32,48,64]")
print("• Training configuration: AdamW optimizer, CosineAnnealingLR scheduler")
print("• Loss function: BCE + Dice (simple combined loss)")
print("• Training: 250 epochs, batch size 8, learning rate 1e-3 → 1e-5")

# Create a second plot showing performance differences
fig2, ax2 = plt.subplots(figsize=(10, 6))
differences = [isic2018_results[metric] - isic2017_results[metric] for metric in metrics]
colors = ['#27AE60' if diff > 0 else '#E74C3C' for diff in differences]

bars = ax2.bar(metrics, differences, color=colors, alpha=0.7)

# Add value labels
for bar, diff in zip(bars, differences):
    height = bar.get_height()
    ax2.annotate(f'{diff:+.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height > 0 else -15),
                textcoords="offset points",
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')

ax2.set_ylabel('Performance Difference (%)', fontsize=12, fontweight='bold')
ax2.set_title('ISIC2017 vs ISIC2018 Performance Differences (UltraLight-VM-UNet)\n(Positive = ISIC2018 Better)', 
              fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('ultralight_vm_unet_isic_performance_differences.png', dpi=300, bbox_inches='tight', facecolor='white')
print("📊 Performance difference plot saved as 'ultralight_vm_unet_isic_performance_differences.png')")
plt.show()

# Create a comprehensive comparison with baseline models
# ESEUNet results for comparison
eseunet_isic2017 = {
    'mIoU': 82.37,
    'DSC': 89.21,
    'Sensitivity': 88.86,
    'Specificity': 98.58
}

eseunet_isic2018 = {
    'mIoU': 80.12,
    'DSC': 87.60,
    'Sensitivity': 87.14,
    'Specificity': 97.55
}

# MSGU-Net baseline for comparison
msgu_isic2017 = {
    'mIoU': 56.11,
    'DSC': 67.90,
    'Sensitivity': 65.63,
    'Specificity': 96.37
}

msgu_isic2018 = {
    'mIoU': 57.95,
    'DSC': 69.85,
    'Sensitivity': 67.74,
    'Specificity': 95.39
}

# RepGhostUNet for comparison
rg_isic2017 = {
    'mIoU': 79.44,
    'DSC': 87.23,
    'Sensitivity': 86.94,
    'Specificity': 98.39
}

rg_isic2018 = {
    'mIoU': 79.73,
    'DSC': 87.43,
    'Sensitivity': 87.35,
    'Specificity': 97.51
}

# Create model comparison plot
fig3, ax3 = plt.subplots(figsize=(14, 8))

x_pos = np.arange(len(metrics))
width = 0.2

# Get average performance for each model
msgu_2017_avg = np.mean(list(msgu_isic2017.values()))
msgu_2018_avg = np.mean(list(msgu_isic2018.values()))
rg_2017_avg = np.mean(list(rg_isic2017.values()))
rg_2018_avg = np.mean(list(rg_isic2018.values()))
eseunet_2017_avg = np.mean(list(eseunet_isic2017.values())) if any(eseunet_isic2017.values()) else 0
eseunet_2018_avg = np.mean(list(eseunet_isic2018.values())) if any(eseunet_isic2018.values()) else 0

bars1 = ax3.bar(x_pos - 1.5*width, list(msgu_isic2017.values()), width, label='MSGU-Net (ISIC2017)', color='#3498DB', alpha=0.8)
bars2 = ax3.bar(x_pos - 0.5*width, list(msgu_isic2018.values()), width, label='MSGU-Net (ISIC2018)', color='#5DADE2', alpha=0.8)
bars3 = ax3.bar(x_pos + 0.5*width, list(rg_isic2017.values()), width, label='RepGhostUNet (ISIC2017)', color='#E74C3C', alpha=0.8)
bars4 = ax3.bar(x_pos + 1.5*width, list(rg_isic2018.values()), width, label='RepGhostUNet (ISIC2018)', color='#EC7063', alpha=0.8)

# Add value labels
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

ax3.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax3.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax3.set_title('Model Comparison: MSGU-Net vs RepGhostUNet', fontsize=14, fontweight='bold', pad=20)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics, fontsize=11)
ax3.legend(fontsize=10, loc='lower right')
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 105)
ax3.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('repghost_unet_vs_msgunet_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("📊 Model comparison plot saved as 'repghost_unet_vs_msgunet_comparison.png'")
plt.show()

# Print comprehensive analysis
print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON: ESEUNet vs RepGhostUNet vs MSGU-Net")
print("="*80)
print("\n📊 Average Performance Scores:")
print(f"  MSGU-Net (ISIC2017):         {msgu_2017_avg:.2f}%")
print(f"  MSGU-Net (ISIC2018):         {msgu_2018_avg:.2f}%")
print(f"  RepGhostUNet (ISIC2017):     {rg_2017_avg:.2f}%")
print(f"  RepGhostUNet (ISIC2018):     {rg_2018_avg:.2f}%")
if eseunet_2017_avg > 0:
    print(f"  ESEUNet (ISIC2017):          {eseunet_2017_avg:.2f}%")
    print(f"  ESEUNet (ISIC2018):          {eseunet_2018_avg:.2f}%")
else:
    print(f"  ESEUNet (ISIC2017):          [Awaiting training results]")
    print(f"  ESEUNet (ISIC2018):          [Awaiting training results]")

print(f"\n🚀 RepGhostUNet Improvement over MSGU-Net:")
print(f"  ISIC2017:  +{rg_2017_avg - msgu_2017_avg:.2f}% ({((rg_2017_avg / msgu_2017_avg - 1) * 100):.1f}% relative)")
print(f"  ISIC2018:  +{rg_2018_avg - msgu_2018_avg:.2f}% ({((rg_2018_avg / msgu_2018_avg - 1) * 100):.1f}% relative)")

print("\n" + "="*80)
