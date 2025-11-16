import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

print("=" * 80)
print("COMPARING DATA REDUCTION METHODS")
print("=" * 80)

# Load all summaries
methods = ['FILTERED', 'EMBEDDED', 'PCA']
summaries = []

for method in methods:
    try:
        df = pd.read_csv(f'Results/{method}/summary.csv')
        summaries.append(df)
    except FileNotFoundError:
        print(f"⚠️ Warning: {method} results not found. Please run train_{method}.py first.")

if not summaries:
    print("❌ No results found. Please run the training scripts first.")
    exit()

# Combine all results
comparison = pd.concat(summaries, ignore_index=True)

# Create comparison output folder
os.makedirs('Results/Comparison', exist_ok=True)
comparison.to_csv('Results/Comparison/comparison_summary.csv', index=False)

print("\n📊 Comparison Summary:")
print(comparison.to_string(index=False))

# Visualization
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Comparison of Data Reduction Methods for Heart Disease Prediction', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. Number of Features
ax1 = plt.subplot(2, 3, 1)
colors = ['steelblue', 'forestgreen', 'purple']
bars = ax1.bar(comparison['Method'], comparison['Features'], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
ax1.set_title('A. Feature Count by Method', fontsize=12, fontweight='bold', loc='left')
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# 2. Test Performance Metrics (Bar Chart)
ax2 = plt.subplot(2, 3, 2)
metrics = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_AUC']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
x = np.arange(len(metrics))
width = 0.25

for i, method in enumerate(comparison['Method']):
    values = comparison.loc[comparison['Method'] == method, metrics].values[0]
    offset = (i - 1) * width
    ax2.bar(x + offset, values, width, label=method, color=colors[i], alpha=0.7, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Metrics', fontsize=11, fontweight='bold')
ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
ax2.set_title('B. Test Performance Comparison', fontsize=12, fontweight='bold', loc='left')
ax2.set_xticks(x)
ax2.set_xticklabels(metric_labels)
ax2.legend(fontsize=10)
ax2.set_ylim([0, 1.1])
ax2.grid(axis='y', alpha=0.3)

# 3. Test Accuracy Comparison (Horizontal Bar)
ax3 = plt.subplot(2, 3, 3)
bars = ax3.barh(comparison['Method'], comparison['Test_Accuracy'], 
                color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_xlabel('Test Accuracy', fontsize=11, fontweight='bold')
ax3.set_title('C. Test Accuracy by Method', fontsize=12, fontweight='bold', loc='left')
ax3.set_xlim([0, 1])
ax3.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
            f'{width:.4f} ({width*100:.2f}%)',
            ha='left', va='center', fontweight='bold', fontsize=10)

# 4. Features vs Accuracy Trade-off
ax4 = plt.subplot(2, 3, 4)
scatter = ax4.scatter(comparison['Features'], comparison['Test_Accuracy'], 
                     c=colors[:len(comparison)], s=400, alpha=0.6, edgecolor='black', linewidth=2)
ax4.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
ax4.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
ax4.set_title('D. Features vs Accuracy Trade-off', fontsize=12, fontweight='bold', loc='left')
ax4.grid(True, alpha=0.3)

for i, method in enumerate(comparison['Method']):
    ax4.annotate(f"{method}\n({int(comparison['Features'].iloc[i])} features)", 
                (comparison['Features'].iloc[i], comparison['Test_Accuracy'].iloc[i]),
                xytext=(10, 10), textcoords='offset points', 
                fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))

# 5. Detailed Metrics Comparison (Radar Chart)
ax5 = plt.subplot(2, 3, 5, projection='polar')
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
num_vars = len(categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

for i, method in enumerate(comparison['Method']):
    values = [
        comparison.loc[comparison['Method'] == method, 'Test_Accuracy'].values[0],
        comparison.loc[comparison['Method'] == method, 'Test_Precision'].values[0],
        comparison.loc[comparison['Method'] == method, 'Test_Recall'].values[0],
        comparison.loc[comparison['Method'] == method, 'Test_F1'].values[0],
        comparison.loc[comparison['Method'] == method, 'Test_AUC'].values[0]
    ]
    values += values[:1]
    ax5.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
    ax5.fill(angles, values, alpha=0.15, color=colors[i])

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories, fontsize=10)
ax5.set_ylim(0, 1)
ax5.set_title('E. Performance Radar Chart', fontsize=12, fontweight='bold', loc='left', pad=20)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. Train vs Test Accuracy
ax6 = plt.subplot(2, 3, 6)
train_accs = comparison['Train_Accuracy'].values
test_accs = comparison['Test_Accuracy'].values
x_pos = np.arange(len(methods))
width = 0.35

bars1 = ax6.bar(x_pos - width/2, train_accs, width, label='Train Accuracy', 
                color='lightblue', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax6.bar(x_pos + width/2, test_accs, width, label='Test Accuracy', 
                color='lightcoral', alpha=0.7, edgecolor='black', linewidth=1.5)

ax6.set_xlabel('Method', fontsize=11, fontweight='bold')
ax6.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax6.set_title('F. Train vs Test Accuracy', fontsize=12, fontweight='bold', loc='left')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(comparison['Method'])
ax6.legend(fontsize=10)
ax6.set_ylim([0, 1.1])
ax6.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('Results/Comparison/comparison_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Comparison completed!")
print("\n📁 Results saved:")
print("   - Results/Comparison/comparison_summary.csv")
print("   - Results/Comparison/comparison_visualization.png")
print("=" * 80)

# Detailed Analysis
print("\n📊 DETAILED ANALYSIS:")
print("=" * 80)

# Best method for each metric
best_accuracy = comparison.loc[comparison['Test_Accuracy'].idxmax()]
best_precision = comparison.loc[comparison['Test_Precision'].idxmax()]
best_recall = comparison.loc[comparison['Test_Recall'].idxmax()]
best_f1 = comparison.loc[comparison['Test_F1'].idxmax()]
best_auc = comparison.loc[comparison['Test_AUC'].idxmax()]
best_features = comparison.loc[comparison['Features'].idxmin()]

print("\n🏆 Best Performance by Metric:")
print(f"   • Accuracy:  {best_accuracy['Method']} ({best_accuracy['Test_Accuracy']:.4f})")
print(f"   • Precision: {best_precision['Method']} ({best_precision['Test_Precision']:.4f})")
print(f"   • Recall:    {best_recall['Method']} ({best_recall['Test_Recall']:.4f})")
print(f"   • F1-Score:  {best_f1['Method']} ({best_f1['Test_F1']:.4f})")
print(f"   • AUC-ROC:   {best_auc['Method']} ({best_auc['Test_AUC']:.4f})")
print(f"   • Fewest Features: {best_features['Method']} ({int(best_features['Features'])} features)")

# Overfitting analysis
print("\n📉 Overfitting Analysis (Train - Test Accuracy):")
for idx, row in comparison.iterrows():
    overfit = row['Train_Accuracy'] - row['Test_Accuracy']
    status = "✅ Good" if overfit < 0.05 else ("⚠️ Moderate" if overfit < 0.10 else "❌ High")
    print(f"   • {row['Method']}: {overfit:.4f} ({overfit*100:.2f}%) {status}")

# Efficiency analysis
print("\n⚡ Efficiency Analysis (Features vs Performance):")
for idx, row in comparison.iterrows():
    efficiency = row['Test_Accuracy'] / row['Features']
    print(f"   • {row['Method']}: {efficiency:.6f} (accuracy per feature)")

# Recommendation
print("\n💡 RECOMMENDATIONS:")
print("=" * 80)

avg_accuracy = comparison['Test_Accuracy'].mean()
best_overall = comparison.loc[comparison['Test_Accuracy'].idxmax()]

print(f"\n1. Best Overall Performance:")
print(f"   → {best_overall['Method']}")
print(f"   → Accuracy: {best_overall['Test_Accuracy']:.4f}")
print(f"   → F1-Score: {best_overall['Test_F1']:.4f}")
print(f"   → Features: {int(best_overall['Features'])}")

print(f"\n2. Most Efficient (Best accuracy-to-features ratio):")
efficiencies = comparison['Test_Accuracy'] / comparison['Features']
most_efficient = comparison.loc[efficiencies.idxmax()]
print(f"   → {most_efficient['Method']}")
print(f"   → {int(most_efficient['Features'])} features with {most_efficient['Test_Accuracy']:.4f} accuracy")

print(f"\n3. Best for Interpretability:")
if comparison.loc[comparison['Method'] == 'FILTERED'].shape[0] > 0:
    filtered_data = comparison.loc[comparison['Method'] == 'FILTERED'].iloc[0]
    print(f"   → FILTERED (original feature names preserved)")
    print(f"   → {int(filtered_data['Features'])} interpretable features")
elif comparison.loc[comparison['Method'] == 'EMBEDDED'].shape[0] > 0:
    embedded_data = comparison.loc[comparison['Method'] == 'EMBEDDED'].iloc[0]
    print(f"   → EMBEDDED (feature importance-based)")
    print(f"   → {int(embedded_data['Features'])} selected features")

print("\n" + "=" * 80)

