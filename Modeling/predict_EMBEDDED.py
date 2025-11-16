import pandas as pd
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# ============================================
# PREDICTION: EMBEDDED Dataset
# ============================================

print("=" * 80)
print("PREDICTION: EMBEDDED Dataset (Embedded Method)")
print("=" * 80)

# Load model and feature names
model = joblib.load('Models/EMBEDDED/model.pkl')
feature_names = joblib.load('Models/EMBEDDED/feature_names.pkl')

# Load test data
X_test = pd.read_csv('Data/test_EMBEDDED/X_test.csv')

# Ensure column order
if list(X_test.columns) != feature_names:
    X_test = X_test[feature_names]

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Save predictions
os.makedirs('Results/EMBEDDED/Predictions', exist_ok=True)
results = pd.DataFrame({
    'Patient_ID': range(1, len(predictions) + 1),
    'Predicted_Class': predictions,
    'Predicted_Label': ['Disease' if p == 1 else 'No Disease' for p in predictions],
    'Probability_No_Disease': probabilities[:, 0],
    'Probability_Disease': probabilities[:, 1]
})
results.to_csv('Results/EMBEDDED/Predictions/predictions.csv', index=False)

print(f"\nEMBEDDED - Predictions completed: {len(predictions)} samples")
print(f"  Disease: {sum(predictions == 1)} ({sum(predictions == 1)/len(predictions)*100:.1f}%)")
print(f"  No Disease: {sum(predictions == 0)} ({sum(predictions == 0)/len(predictions)*100:.1f}%)")

# Evaluate if labels available
try:
    y_test = pd.read_csv('Data/test_EMBEDDED/y_test.csv')['num'].values
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities[:, 1])
    
    print(f"\nEMBEDDED - Evaluation Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]:3d}  FP: {cm[0][1]:3d}")
    print(f"  FN: {cm[1][0]:3d}  TP: {cm[1][1]:3d}")
    
    # Save evaluation
    eval_results = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Score': [accuracy, precision, recall, f1, auc]
    })
    eval_results.to_csv('Results/EMBEDDED/Predictions/evaluation_results.csv', index=False)
    
    # Detailed comparison
    comparison = pd.DataFrame({
        'Patient_ID': range(1, len(predictions) + 1),
        'Actual': y_test,
        'Actual_Label': ['Disease' if y == 1 else 'No Disease' for y in y_test],
        'Predicted': predictions,
        'Predicted_Label': ['Disease' if p == 1 else 'No Disease' for p in predictions],
        'Correct': y_test == predictions,
        'Probability_Disease': probabilities[:, 1]
    })
    comparison.to_csv('Results/EMBEDDED/Predictions/prediction_comparison.csv', index=False)
    
    print(f"\nCorrect: {sum(y_test == predictions)}/{len(predictions)}")
    
    # Display detailed predictions
    print("\n" + "="*90)
    print(f"EMBEDDED - DETAILED PREDICTIONS")
    print("="*90)
    print(f"{'ID':<5} {'Truth':<12} {'Predicted':<12} {'Prob(Disease)':<15} {'Status':<10}")
    print("-"*90)
    
    n_display = min(20, len(predictions))
    for i in range(n_display):
        patient_id = i + 1
        truth_label = 'Disease' if y_test[i] == 1 else 'No Disease'
        pred_label = 'Disease' if predictions[i] == 1 else 'No Disease'
        prob_disease = probabilities[i, 1]
        status = 'CORRECT' if y_test[i] == predictions[i] else 'WRONG'
        
        print(f"{patient_id:<5} {truth_label:<12} {pred_label:<12} {prob_disease:<15.4f} {status:<10}")
    
    if len(predictions) > n_display:
        print(f"... ({len(predictions) - n_display} more samples)")
    print("="*90)
    
    # Summary of errors
    errors = comparison[comparison['Correct'] == False]
    if len(errors) > 0:
        print(f"\nIncorrect Predictions ({len(errors)} cases):")
        print("-"*90)
        for idx, row in errors.head(10).iterrows():
            print(f"Patient {row['Patient_ID']}: Truth={row['Actual_Label']}, Predicted={row['Predicted_Label']}, Prob={row['Probability_Disease']:.4f}")
    else:
        print("\nAll predictions are correct!")
    
except FileNotFoundError:
    print("\nNo ground truth labels found. Evaluation skipped.")

print(f"\nEMBEDDED prediction completed.")
print("=" * 80)

