import os
import json
import pandas as pd
import numpy as np


class OmegaZooProcessor:
    def __init__(self, base_path: str):
        """
        C:\Users\ub13-glab-020\Downloads\AI-MANOJ
        """
        self.base_path = base_path
        self.zoo_df = None
        self.class_df = None
        self.aux_df = None
        self.merged_data = None

    # -----------------------------
    # Helper: normalize animal name
    # -----------------------------
    def _omega_normalize_name_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find the column that has animal names and create a normalized
        'ANIMAL_NAME' column in UPPERCASE for merging.
        """
        possible_name_cols = [
            c for c in df.columns
            if c.strip().lower() in ["name", "animal_name", "animal", "animalname"]
        ]

        if not possible_name_cols:
            raise ValueError(
                f"Could not find an animal name column in columns: {df.columns.tolist()}"
            )

        name_col = possible_name_cols[0]
        df["ANIMAL_NAME"] = (
            df[name_col]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        return df

    # ----------------------------------
    # Helper: clean / standardize JSON
    # ----------------------------------
    def _omega_clean_aux_json(self, raw_records):
        """
        Fix JSON data inconsistencies:
          1) Standardize field names:
             - consevrvation_status, conservation, status -> conservation_status
             - habitat, habitats -> habitat_type
             - diet_type -> diet
          2) Fix typos in diet categories (e.g. 'omnivor' -> 'omnivore')
          3) Standardize habitat values (e.g. 'fresh water' -> 'freshwater')
          4) Normalize animal names to ANIMAL_NAME (UPPERCASE)
        """
        cleaned_records = []

        for rec in raw_records:
            if not isinstance(rec, dict):
                # Skip corrupted / non-dict entries
                continue

            r = dict(rec)  # shallow copy to avoid mutating original

            # 1) standardize field names
            key_mapping = {
                "consevrvation_status": "conservation_status",
                "conservation": "conservation_status",
                "status": "conservation_status",
                "habitat": "habitat_type",
                "habitats": "habitat_type",
                "diet_type": "diet",
            }

            for old_key, new_key in key_mapping.items():
                if old_key in r and new_key not in r:
                    r[new_key] = r.pop(old_key)

            # 2) fix typos in diet categories
            diet = str(r.get("diet", "")).strip().lower()
            if diet:
                if diet in ["omnivor", "omniver", "omnivorous"]:
                    diet = "omnivore"
                elif diet in ["herbavor", "herbivor"]:
                    diet = "herbivore"
                elif diet in ["carnvor", "carnivor"]:
                    diet = "carnivore"
                r["diet"] = diet

            # 3) standardize habitat values
            habitat = str(r.get("habitat_type", "")).strip().lower()
            if habitat:
                habitat = habitat.replace("fresh water", "freshwater")
                habitat = habitat.replace("rain forest", "rainforest")
                habitat = habitat.replace("grass land", "grassland")
                r["habitat_type"] = habitat

            # 4) normalize animal name to ANIMAL_NAME (UPPERCASE)
            name_keys = [k for k in r.keys() if k.lower() in ["name", "animal_name", "animal"]]
            if name_keys:
                nk = name_keys[0]
                r["ANIMAL_NAME"] = str(r[nk]).strip().upper()

            cleaned_records.append(r)

        return cleaned_records

    # -----------------------------------------
    # MAIN TASK 1 METHOD: omega_load_and_integrate
    # -----------------------------------------
    def omega_load_and_integrate(self):
        """
        TASK 1: Data loading and integration
          A) Load datasets
          B) Normalize names
          C) Fix JSON inconsistencies
          D) Merge all datasets
          E) Handle missing values
          F) Create two custom biological features
        """
        # ------------ A) Load all three datasets ------------
        zoo_path = os.path.join(self.base_path, "zoo.csv")
        class_path = os.path.join(self.base_path, "class.csv")
        aux_path = os.path.join(self.base_path, "auxiliary_metadata.json")

        # 1) load zoo.csv with proper encoding
        self.zoo_df = pd.read_csv(zoo_path, encoding="utf-8")

        # 2) load class.csv with appropriate method
        self.class_df = pd.read_csv(class_path)

        # 3) load auxiliary_metadata.json handling corrupted fields
        with open(aux_path, "r", encoding="utf-8") as f:
            try:
                raw_aux = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Error reading JSON file: {e}")

        # C) fix JSON data inconsistencies (field names, diet, habitat, names)
        cleaned_aux = self._omega_clean_aux_json(raw_aux)
        self.aux_df = pd.DataFrame(cleaned_aux)

        # ------------ B) Name normalization to UPPERCASE ------------
        self.zoo_df = self._omega_normalize_name_column(self.zoo_df)
        self.class_df = self._omega_normalize_name_column(self.class_df)

        if "ANIMAL_NAME" not in self.aux_df.columns:
            self.aux_df = self._omega_normalize_name_column(self.aux_df)

        # ------------ D) Merge all datasets ------------
        # Primary dataset = zoo_df (no data loss from zoo_df)
        # Merge on animal names after normalization (ANIMAL_NAME)

        # Merge zoo + class
        merged = pd.merge(
            self.zoo_df,
            self.class_df.drop_duplicates(subset=["ANIMAL_NAME"]),
            on="ANIMAL_NAME",
            how="left",              # left join to keep all zoo animals
            suffixes=("", "_CLASS"),
        )

        # Merge (zoo+class) + auxiliary
        merged = pd.merge(
            merged,
            self.aux_df.drop_duplicates(subset=["ANIMAL_NAME"]),
            on="ANIMAL_NAME",
            how="left",              # left join: animals not in aux -> NaN in aux columns
            suffixes=("", "_AUX"),
        )

        # ------------ E) Handle missing values ------------
        # - Categorical (object / category) -> "unknown"
        # - Numerical -> median of that column
        numeric_cols = merged.select_dtypes(include=["number"]).columns
        categorical_cols = merged.columns.difference(numeric_cols)

        # Fill numeric NaN with median
        for col in numeric_cols:
            if merged[col].isnull().any():
                median_val = merged[col].median()
                merged[col] = merged[col].fillna(median_val)

        # Fill categorical NaN with "unknown"
        for col in categorical_cols:
            if merged[col].isnull().any():
                merged[col] = merged[col].fillna("unknown")

        # ------------ F) Create two custom biological features ------------
        engineered_feature_names = []

        # Feature 1: is_mammal_like
        # If the animal has 'milk' == 1, we consider it biologically mammal-like.
        # This is a common rule in biology: mammals are characterized by milk production.
        if "milk" in merged.columns:
            merged["is_mammal_like"] = (merged["milk"] > 0).astype(int)
            engineered_feature_names.append("is_mammal_like")

        # Feature 2: mobility_score
        # Combine aquatic & airborne capabilities + number of legs.
        # Idea:
        #   - aquatic * 2  : swimming ability
        #   - airborne * 2 : flying ability
        #   - legs / 2     : more legs often => better movement on land
        # This gives a simple composite "mobility" feature.
        if "aquatic" in merged.columns:
            aquatic = merged["aquatic"].astype(float)
            airborne = merged["airborne"].astype(float) if "airborne" in merged.columns else 0.0
            if "legs" in merged.columns:
                legs = merged["legs"].astype(float)
                merged["mobility_score"] = aquatic * 2 + airborne * 2 + legs / 2.0
            else:
                merged["mobility_score"] = aquatic * 2 + airborne * 2

            engineered_feature_names.append("mobility_score")

        self.merged_data = merged

        # ------------ Required Outputs ------------
        print(f"Dataset shape : {self.merged_data.shape}")

        missing_per_col = self.merged_data.isnull().sum()
        total_missing = int(missing_per_col.sum())
        print(f"missing values: (per_column={dict(missing_per_col)}, total={total_missing})")

        print(f"Duplicate rows : {self.merged_data.duplicated().sum()}")

        print("\nFirst 3 rows:")
        print(self.merged_data.head(3))

        print(f"\nEngineered features: {list(engineered_feature_names)}")

        return self.merged_data, engineered_feature_names


# --------------------------
# Example usage (for you)
# --------------------------
if __name__ == "__main__":
    base_path = r"C:\Users\ub13-glab-020\Downloads\AI-MANOJ"
    processor = OmegaZooProcessor(base_path)
    processor.omega_load_and_integrate()
