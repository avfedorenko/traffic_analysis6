from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


class ResumeSeniorityClassifier:
    """Classifier for determining developer seniority level (junior/middle/senior)."""

    def __init__(self, features_path: str, labels_path: str):
        """Initialize the classifier.

        Args:
            features_path: Path to the features file (x_data.npy)
            labels_path: Path to the labels file (y_data.npy)
        """
        self.features_path = features_path
        self.labels_path = labels_path
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self._is_fitted = False

    def load_arrays(self) -> None:
        """Load preprocessed data from npy files."""
        print("Loading preprocessed data...")
        self.X = np.load(self.features_path)
        self.y = np.load(self.labels_path, allow_pickle=True)
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")

    def save_label_distribution_plot(self, output_dir: Path) -> None:
        """Plot class distribution and save it to disk."""
        print("\nPlotting class distribution...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Count plot
        unique, counts = np.unique(self.y, return_counts=True)
        class_counts = pd.Series(counts, index=unique)

        axes[0].bar(
            class_counts.index,
            class_counts.values,
            color=["#3498db", "#2ecc71", "#e74c3c"],
        )
        axes[0].set_xlabel("Developer Level")
        axes[0].set_ylabel("Number of Resumes")
        axes[0].set_title("Distribution of Developer Levels")
        axes[0].grid(axis="y", alpha=0.3)

        # Add count labels
        for i, (level, count) in enumerate(class_counts.items()):
            axes[0].text(i, count, str(count), ha="center", va="bottom")

        # Pie chart
        colors = ["#3498db", "#2ecc71", "#e74c3c"]
        axes[1].pie(
            class_counts.values,
            labels=class_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        axes[1].set_title("Percentage Distribution")

        plt.tight_layout()
        output_path = output_dir / "class_distribution.png"
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to: {output_path}")
        plt.close()

    def split_dataset(self) -> None:
        """Prepare data for training (train/test split)."""
        print("\nSplitting data into train/test sets...")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )

        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")

    def fit_model(self) -> None:
        """Train the classifier."""
        print("\nTraining Random Forest classifier...")

        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
        )

        self.model.fit(self.X_train, self.y_train)
        self._is_fitted = True
        print("Model trained successfully")

    def evaluate_model(self, output_dir: Path) -> dict:
        """Evaluate the model and save visualizations."""
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)

        y_pred = self.model.predict(self.X_test)

        print("\n", classification_report(self.y_test, y_pred, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(
            self.y_test, y_pred, labels=["junior", "middle", "senior"]
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["junior", "middle", "senior"],
            yticklabels=["junior", "middle", "senior"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        cm_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300)
        print(f"\nConfusion matrix saved to: {cm_path}")
        plt.close()

        # Feature importance
        feature_names = [
            "age",
            "salary",
            "experience_years",
            "city",
            "education",
            "employment",
            "has_car",
        ]
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE")
        print("=" * 70)
        for i, idx in enumerate(indices):
            print(f"{i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(
            range(len(importances)), [feature_names[i] for i in indices], rotation=45
        )
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        fi_path = output_dir / "feature_importance.png"
        plt.savefig(fi_path, dpi=300)
        print("\nFeature importance plot saved to:")
        print(fi_path)
        plt.close()

        accuracy = accuracy_score(self.y_test, y_pred)
        f1_macro = f1_score(self.y_test, y_pred, average="macro")

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "y_pred": y_pred,
        }

    def print_metrics_summary(self, metrics: dict) -> None:
        """Print key metrics summary."""
        print(f"\nAccuracy: {metrics['accuracy']:.3f}")
        print(f"Macro F1: {metrics['f1_macro']:.3f}")

    def run(self, output_dir: Optional[Path] = None) -> None:
        """Run the complete classification pipeline."""
        if output_dir is None:
            output_dir = Path("images") / "png"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.load_arrays()
        self.save_label_distribution_plot(output_dir)
        self.split_dataset()
        self.fit_model()
        metrics = self.evaluate_model(output_dir)
        self.print_metrics_summary(metrics)
