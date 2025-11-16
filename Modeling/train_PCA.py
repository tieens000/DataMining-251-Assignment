import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# DATASET: PCA (Principal Component Analysis)
# ============================================

# Load data
df = pd.read_csv('D:\\Data_Mining\\BTL\\DataMining-251-Assignment-main\\Process\\Reduced_data\\reduced_heart_disease_PCA.csv')

# Split features and target
X = df.drop('num', axis=1)
y = df['num']

# Train-test split 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save split data
os.makedirs('Data/train_PCA', exist_ok=True)
os.makedirs('Data/test_PCA', exist_ok=True)
X_train.to_csv('Data/train_PCA/X_train.csv', index=False)
y_train.to_csv('Data/train_PCA/y_train.csv', index=False)
X_test.to_csv('Data/test_PCA/X_test.csv', index=False)
y_test.to_csv('Data/test_PCA/y_test.csv', index=False)

print("=" * 80)
print("TRAINING: PCA Dataset (Principal Component Analysis - 95% Variance)")
print("=" * 80)
print(f"Features: {X.shape[1]} principal components")
print(f"Samples: {X.shape[0]} rows")
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Train model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

# Evaluate
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"\nTrain Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy:  {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1-Score:  {test_f1:.4f}")
print(f"Test AUC-ROC:   {test_auc:.4f}")

# Create output folders
os.makedirs('Results/PCA/Training', exist_ok=True)

# Feature (Component) importance
feature_importance = pd.DataFrame({
    'Component': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
feature_importance.to_csv('Results/PCA/Training/component_importance.csv', index=False)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Random Forest Training Results - PCA Dataset', fontsize=16, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[0, 0],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# Component Importance
top_features = feature_importance.head(15)
axes[0, 1].barh(top_features['Component'], top_features['Importance'], color='purple')
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Top 15 Principal Components')
axes[0, 1].invert_yaxis()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
axes[1, 0].plot(fpr, tpr, color='purple', lw=2, label=f'AUC = {test_auc:.2f}')
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Train vs Test
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
train_scores = [train_accuracy, precision_score(y_train, y_train_pred), 
                recall_score(y_train, y_train_pred), f1_score(y_train, y_train_pred),
                roc_auc_score(y_train, rf_model.predict_proba(X_train)[:, 1])]
test_scores = [test_accuracy, test_precision, test_recall, test_f1, test_auc]

x = np.arange(len(metrics))
width = 0.35
axes[1, 1].bar(x - width/2, train_scores, width, label='Train', color='plum')
axes[1, 1].bar(x + width/2, test_scores, width, label='Test', color='purple')
axes[1, 1].set_xlabel('Metrics')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Performance')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics)
axes[1, 1].legend()
axes[1, 1].set_ylim([0, 1.1])
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('Results/PCA/Training/training_results.png', dpi=300, bbox_inches='tight')
plt.close()

# Save model
os.makedirs('Models/PCA', exist_ok=True)
joblib.dump(rf_model, 'Models/PCA/model.pkl')
joblib.dump(list(X_train.columns), 'Models/PCA/feature_names.pkl')

# Save summary
summary = {
    'Method': 'PCA',
    'Description': 'Principal Component Analysis (95% Variance)',
    'Features': X.shape[1],
    'Train_Accuracy': train_accuracy,
    'Test_Accuracy': test_accuracy,
    'Test_Precision': test_precision,
    'Test_Recall': test_recall,
    'Test_F1': test_f1,
    'Test_AUC': test_auc
}
pd.DataFrame([summary]).to_csv('Results/PCA/summary.csv', index=False)

print("\n✅ Training completed!")
print(f"   📁 Model saved: Models/PCA/")
print(f"   📁 Results saved: Results/PCA/Training/")
print("=" * 80)

