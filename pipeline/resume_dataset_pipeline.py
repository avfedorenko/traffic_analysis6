import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class ResumeDatasetPipeline:
    """Preprocessing pipeline for resume-based seniority classification."""

    def __init__(self, csv_path: str):
        """Initialize the pipeline.

        Args:
            csv_path: Path to the CSV file with resume data.
        """
        self.csv_path = csv_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.df_developers: pd.DataFrame = pd.DataFrame()
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def run(self, output_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Run the complete preprocessing pipeline.

        Args:
            output_dir: Directory to save processed data.

        Returns:
            Tuple of (X, y) arrays.
        """
        if output_dir is None:
            output_dir = Path(self.csv_path).parent

        self.load_csv()
        self.normalize_dataset()
        self.select_developer_rows()
        self.derive_seniority_label()
        self.build_feature_columns()
        self.collapse_rare_categories()
        self.filter_outliers_iqr()
        X, y = self.build_training_arrays()

        # Save to npy files
        x_path = output_dir / "x_data.npy"
        y_path = output_dir / "y_data.npy"

        np.save(x_path, X)
        np.save(y_path, y)

        print("\nData saved:")
        print(f"  X: {x_path} (shape: {X.shape})")
        print(f"  y: {y_path} (shape: {y.shape})")

        return X, y

    def load_csv(self) -> None:
        """Load data from CSV file."""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        print(f"Total resumes loaded: {len(self.df)}")

    def normalize_dataset(self) -> None:
        """Clean data by removing duplicates and normalizing text."""
        print("\nCleaning data...")
        initial_count = len(self.df)

        # Remove duplicates
        self.df = self.df.drop_duplicates()
        duplicates_removed = initial_count - len(self.df)
        print(f"Removed {duplicates_removed} duplicates")

        # Clean text columns
        text_columns = self.df.select_dtypes(include=["object"]).columns
        for column in text_columns:
            if column in self.df.columns:
                self.df[column] = self.df[column].astype(str)
                self.df[column] = self.df[column].str.replace("\ufeff", "", regex=False)
                self.df[column] = self.df[column].str.replace("\xa0", " ", regex=False)
                self.df[column] = self.df[column].str.replace(r"[\t\n\r]", " ", regex=True)
                self.df[column] = (
                    self.df[column].str.replace(r"\s+", " ", regex=True).str.strip()
                )

        print(f"Data cleaned: {len(self.df)} rows remaining")

    def select_developer_rows(self) -> None:
        """Filter IT developers from the dataset."""
        print("\nFiltering IT developers...")

        developer_keywords = [
            r"разработчик",
            r"programmer",
            r"developer",
            r"программист",
            r"python",
            r"java(?!script)",
            r"\.net",
            r"c\+\+",
            r"php",
            r"frontend",
            r"backend",
            r"fullstack",
            r"full stack",
            r"software engineer",
        ]

        pattern = "|".join(developer_keywords)
        job_title_col = self.df.columns[3]
        mask = self.df[job_title_col].str.contains(
            pattern, case=False, na=False, regex=True
        )

        self.df_developers = self.df[mask].copy()
        print(f"IT developers found: {len(self.df_developers)}")

    def derive_seniority_label(self) -> None:
        """Create target variable (junior/middle/senior) based on title and experience."""
        print("\nCreating target variable...")

        def extract_level(row) -> str:
            job_title = str(row[self.df_developers.columns[3]]).lower()
            experience_text = str(row[self.df_developers.columns[6]]).lower()
            experience_years = self._parse_experience_years(experience_text)

            if any(keyword in job_title for keyword in ["junior", "джуниор", "младший"]):
                return "junior"
            elif any(
                keyword in job_title
                for keyword in [
                    "senior",
                    "сеньор",
                    "старший",
                    "lead",
                    "principal",
                    "architect",
                ]
            ):
                return "senior"
            elif any(keyword in job_title for keyword in ["middle", "миддл", "средний"]):
                return "middle"

            if experience_years is not None:
                if experience_years < 2:
                    return "junior"
                elif experience_years < 5:
                    return "middle"
                else:
                    return "senior"

            return "middle"

        self.df_developers["level"] = self.df_developers.apply(extract_level, axis=1)

        print("\nClass distribution:")
        print(self.df_developers["level"].value_counts())
        percentages = self.df_developers["level"].value_counts(normalize=True) * 100
        print(f"\nPercentages:\n{percentages}")

    @staticmethod
    def _parse_experience_years(text: str) -> Optional[float]:
        """Extract years of experience from text."""
        patterns = [
            r"опыт работы (\d+) (?:год|года|лет)(?: (\d+) месяц)?",
            r"(\d+)\s*(?:year|года|лет)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                years = int(match.group(1))
                months = int(match.group(2)) if match.lastindex > 1 else 0
                return years + months / 12

        return None

    def build_feature_columns(self) -> None:
        """Extract model features from resume data."""
        print("\nExtracting features...")

        self.df_developers["age"] = self.df_developers[self.df_developers.columns[1]].apply(
            self._parse_age
        )

        self.df_developers["salary"] = self.df_developers[
            self.df_developers.columns[2]
        ].apply(self._parse_salary_k)

        self.df_developers["city"] = self.df_developers[self.df_developers.columns[4]].apply(
            self._parse_city
        )

        self.df_developers["experience_years"] = self.df_developers[
            self.df_developers.columns[6]
        ].apply(self._parse_experience_years)

        self.df_developers["education"] = self.df_developers[
            self.df_developers.columns[9]
        ].apply(self._parse_education)

        self.df_developers["employment"] = self.df_developers[
            self.df_developers.columns[5]
        ].apply(lambda x: "full" if "полная" in str(x).lower() else "part")

        self.df_developers["has_car"] = self.df_developers[self.df_developers.columns[11]].apply(
            lambda x: (
                1
                if "автомобиль" in str(x).lower() and "не указано" not in str(x).lower()
                else 0
            )
        )

        # Fill missing values
        self.df_developers["age"].fillna(self.df_developers["age"].median(), inplace=True)
        self.df_developers["salary"].fillna(
            self.df_developers["salary"].median(), inplace=True
        )
        self.df_developers["experience_years"].fillna(
            self.df_developers["experience_years"].median(), inplace=True
        )
        self.df_developers["city"].fillna("Unknown", inplace=True)
        self.df_developers["education"].fillna("secondary", inplace=True)

        print("Features extracted successfully")

    def filter_outliers_iqr(self) -> None:
        """Remove outliers using IQR method for numerical features."""
        print("\nRemoving outliers...")
        initial_count = len(self.df_developers)

        numerical_cols = ["age", "salary", "experience_years"]

        for col in numerical_cols:
            if col in self.df_developers.columns:
                # Skip if no data
                if self.df_developers[col].isna().all():
                    continue

                Q1 = self.df_developers[col].quantile(0.25)
                Q3 = self.df_developers[col].quantile(0.75)
                IQR = Q3 - Q1

                # Use wider bounds (3 * IQR) to be less aggressive
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                before = len(self.df_developers)

                # Remove outliers, handling NaN properly
                mask = (
                    self.df_developers[col].notna()
                    & (self.df_developers[col] >= lower_bound)
                    & (self.df_developers[col] <= upper_bound)
                )
                self.df_developers = self.df_developers[mask]

                after = len(self.df_developers)
                print(
                    f"  {col}: removed {before - after} outliers "
                    f"(bounds: {lower_bound:.1f} - {upper_bound:.1f})"
                )

        outliers_removed = initial_count - len(self.df_developers)
        print(f"Total outliers removed: {outliers_removed}")
        print(f"Remaining developers: {len(self.df_developers)}")

    def collapse_rare_categories(self) -> None:
        """Group rare categories into 'Other' for categorical features."""
        print("\nGrouping rare categories...")

        # For 'city', keep only major cities
        if "city" in self.df_developers.columns:
            city_counts = self.df_developers["city"].value_counts()
            # Keep cities with at least 1% of data
            threshold = len(self.df_developers) * 0.01
            rare_cities = city_counts[city_counts < threshold].index

            initial_unique = self.df_developers["city"].nunique()
            self.df_developers.loc[
                self.df_developers["city"].isin(rare_cities), "city"
            ] = "Other"
            final_unique = self.df_developers["city"].nunique()

            print(
                f"City categories: {initial_unique} → {final_unique} "
                f"({initial_unique - final_unique} grouped into 'Other')"
            )

        # For 'education', already grouped as higher/vocational/secondary

        print("Category grouping completed")

    @staticmethod
    def _parse_age(text: str) -> Optional[float]:
        """Extract age from text."""
        match = re.search(r"(\d+)\s*(?:год|года|лет)", str(text))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_salary_k(text: str) -> Optional[float]:
        """Extract salary from text (in thousands of rubles)."""
        text = str(text).replace(" ", "").replace("\xa0", "")
        match = re.search(r"(\d+)(?:000)?", text)
        if match:
            salary = float(match.group(1))
            return salary if salary < 1000 else salary / 1000
        return None

    @staticmethod
    def _parse_city(text: str) -> str:
        """Extract city name from text."""
        text = str(text)
        match = re.search(r"^([^,]+)", text)
        if match:
            city = match.group(1).strip()
            major_cities = ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург"]
            return city if city in major_cities else "Other"
        return "Unknown"

    @staticmethod
    def _parse_education(text: str) -> str:
        """Extract education level."""
        text = str(text).lower()
        if "высшее" in text:
            return "higher"
        elif "среднее специальное" in text or "техникум" in text:
            return "vocational"
        else:
            return "secondary"

    def build_training_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Encode features, scale them, and return X/y arrays."""
        print("\nPreparing data...")

        feature_cols = [
            "age",
            "salary",
            "experience_years",
            "city",
            "education",
            "employment",
            "has_car",
        ]

        # Remove rows with missing target
        self.df_developers = self.df_developers[self.df_developers["level"].notna()].copy()

        # Encode categorical features
        for col in ["city", "education", "employment"]:
            le = LabelEncoder()
            self.df_developers[col] = le.fit_transform(self.df_developers[col])
            self.label_encoders[col] = le

        # Prepare X and y
        X = self.df_developers[feature_cols].values
        y = self.df_developers["level"].values

        # Scale features
        X = self.scaler.fit_transform(X)

        print(f"Dataset size: {len(X)}")
        print(f"Features: {X.shape[1]}")

        return X, y
