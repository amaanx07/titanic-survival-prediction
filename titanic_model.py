import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')
matplotlib.use('Agg') 
sns.set_theme(style='whitegrid')
os.makedirs('plots', exist_ok=True)

SEED = 42
np.random.seed(SEED)

# =============================================================================
# STEP 1: LOAD YOUR DATA
# =============================================================================
print("\n" + "="*60)
print("🚢 TITANIC SURVIVAL PREDICTION")
print("="*60)

# Straight to the point: Read your uploaded file
data = pd.read_csv('train.csv')
print("\n✅ Successfully loaded 'train.csv'")

# =============================================================================
# STEP 2: EXPLORE THE DATA (EDA)
# =============================================================================
print("\n" + "-"*60)
print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("-"*60)

total_people = len(data)
total_survived = data['Survived'].sum()
survival_rate = data['Survived'].mean()

print(f"\nTotal Passengers: {total_people:,}")
print(f"Survived:         {total_survived} ({survival_rate:.1%})")
print(f"Did Not Survive:  {total_people - total_survived} ({1 - survival_rate:.1%})")

# Draw Charts
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Titanic - Exploratory Data Analysis', fontsize=17, fontweight='bold', y=1.01)

def draw_percentage_bar_chart(ax, categories, values, colors, title):
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    ax.set_title(title, fontweight='bold', fontsize=12, pad=8)
    ax.set_ylabel('Survival Rate', fontsize=11)
    ax.set_ylim(0, 1.15) 
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.4)

# 1. Survival by Gender
gender_survival = data.groupby('Sex')['Survived'].mean()
draw_percentage_bar_chart(axes[0,0], gender_survival.index, gender_survival.values, ['#E74C3C','#3498DB'], 'Survival Rate by Sex')

# 2. Survival by Ticket Class
class_survival = data.groupby('Pclass')['Survived'].mean()
draw_percentage_bar_chart(axes[0,1], [f'Class {c}' for c in class_survival.index], class_survival.values, 
                          ['#F39C12','#27AE60','#8E44AD'], 'Survival Rate by Ticket Class')

# 3. Age Distribution
for survived, label, color in [(0, 'Did Not Survive', '#E74C3C'), (1, 'Survived', '#2ECC71')]:
    axes[1,0].hist(data[data['Survived']==survived]['Age'].dropna(), bins=25, alpha=0.65, color=color, label=label)
axes[1,0].set_title('Age Distribution by Survival', fontweight='bold', fontsize=12)
axes[1,0].legend()

# 4. Survival by Port
port_survival = data.groupby('Embarked')['Survived'].mean().dropna()
draw_percentage_bar_chart(axes[1,1], ['Southampton', 'Cherbourg', 'Queenstown'], port_survival.values,
                          ['#1ABC9C','#E67E22','#9B59B6'], 'Survival Rate by Embarkation Port')

plt.tight_layout()
plt.savefig('plots/01_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Saved plot -> plots/01_eda.png")

# =============================================================================
# STEP 3: PREPARE DATA FOR MACHINE LEARNING
# =============================================================================
print("\n" + "-"*60)
print("STEP 3: PREPARING DATA")
print("-"*60)

def prepare_data(raw_data):
    df = raw_data.copy()

    # Fill in blanks for Age and Fare with the middle values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S') 

    # Calculate family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Drop columns the AI doesn't need
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

    # Convert text (like male/female) into numbers
    for text_column in df.select_dtypes(include=['object', 'category']).columns:
        df[text_column] = pd.factorize(df[text_column])[0]

    return df

clean_data = prepare_data(data)

feature_names = [col for col in clean_data.columns if col != 'Survived']
features = clean_data[feature_names].values
target = clean_data['Survived'].values

print(f"\nUsing {len(feature_names)} clues: {feature_names}")

# Chart: Feature Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(clean_data.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
            mask=np.triu(np.ones_like(clean_data.corr(), dtype=bool)), ax=ax)
ax.set_title('Feature Correlation', fontsize=14, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('plots/02_correlation.png', dpi=150)
plt.close()
print("✅ Saved plot -> plots/02_correlation.png")

# =============================================================================
# STEP 4: TRAIN / TEST SPLIT
# =============================================================================
print("\n" + "-"*60)
print("STEP 4: SPLITTING DATA (80% Train, 20% Test)")
print("-"*60)

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.20, random_state=SEED, stratify=target
)

scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled  = scaler.transform(features_test)

print(f"\nTraining Data: {len(features_train)} passengers")
print(f"Testing Data:  {len(features_test)} passengers")

# =============================================================================
# STEP 5 & 6: TRAINING & EVALUATING
# =============================================================================
print("\n" + "-"*60)
print("STEP 5: TRAINING THE AI MODELS")
print("-"*60)

# Model 1: Logistic Regression
print("\n[1] Training Logistic Regression...")
log_model = LogisticRegression(random_state=SEED)
log_model.fit(features_train_scaled, target_train)

log_predictions = log_model.predict(features_test_scaled)
log_accuracy = accuracy_score(target_test, log_predictions)
log_probabilities = log_model.predict_proba(features_test_scaled)[:, 1]
log_auc = roc_auc_score(target_test, log_probabilities)

# Model 2: Random Forest
print("[2] Training Random Forest...")
forest_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED)
forest_model.fit(features_train, target_train) 

forest_predictions = forest_model.predict(features_test)
forest_accuracy = accuracy_score(target_test, forest_predictions)
forest_probabilities = forest_model.predict_proba(features_test)[:, 1]
forest_auc = roc_auc_score(target_test, forest_probabilities)

# Chart: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, title in zip(axes, [log_predictions, forest_predictions], ['Logistic Regression', 'Random Forest']):
    cm = confusion_matrix(target_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=['Died', 'Survived']).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(title, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('plots/03_confusion_matrices.png', dpi=150)
plt.close()
print("\n✅ Saved plot -> plots/03_confusion_matrices.png")

# Chart: ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
for probs, label, color in [(log_probabilities, f'Logistic Reg. (Score: {log_auc:.3f})', 'red'), 
                            (forest_probabilities, f'Random Forest (Score: {forest_auc:.3f})', 'blue')]:
    fpr, tpr, _ = roc_curve(target_test, probs)
    ax.plot(fpr, tpr, lw=2.5, color=color, label=label)
ax.plot([0,1],[0,1], 'k--', alpha=0.5, label='Random Guessing')
ax.set_title('Model Performance (ROC Curves)', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('plots/04_roc_curves.png', dpi=150)
plt.close()
print("✅ Saved plot -> plots/04_roc_curves.png")

# Chart: Feature Importance
clue_importance = pd.Series(forest_model.feature_importances_, index=feature_names).sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
clue_importance.plot(kind='barh', color='#2ECC71', ax=ax)
ax.set_title('Which features mattered most? (Random Forest)', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/05_feature_importance.png', dpi=150)
plt.close()
print("✅ Saved plot -> plots/05_feature_importance.png")

# =============================================================================
# STEP 7: FINAL REPORT CARD
# =============================================================================
print("\n" + "="*60)
print("📊 FINAL SUMMARY")
print("="*60)

print(f"\n  Logistic Regression -> Accuracy: {log_accuracy:.1%}")
print(f"  Random Forest       -> Accuracy: {forest_accuracy:.1%}")

winner = 'Random Forest' if forest_accuracy > log_accuracy else 'Logistic Regression'
print(f"\n🏆 Best model overall: {winner}")

top_3_clues = list(clue_importance.nlargest(3).index)
print(f"💡 The top 3 predictive features were: {top_3_clues}")

print("\n" + "="*60 + "\n")