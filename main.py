import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import RFECV, RFE
from typing import Tuple, List, Optional, Dict

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PreprocessingHandler:
    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menghasilkan ringkasan statistik deskriptif dari DataFrame.

        Args:
            df (pd.DataFrame): DataFrame input.

        Returns:
            pd.DataFrame: DataFrame berisi statistik deskriptif.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        summary = pd.DataFrame({
            "Miss": numeric_df.isnull().sum(),
            "Min": numeric_df.min(),
            "Max": numeric_df.max(),
            "Mean": numeric_df.mean(),
            "Med": numeric_df.median(),
            "Mode": numeric_df.mode().iloc[0]
        })
        summary.reset_index(inplace=True)
        summary.rename(columns={"index": "Atribut"}, inplace=True)
        summary.insert(0, "No", range(1, len(summary) + 1))
        return summary

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Memuat dataset dari file CSV.

        Args:
            file_path (str): Path ke file CSV.

        Returns:
            pd.DataFrame: DataFrame yang dimuat.

        Raises:
            FileNotFoundError: Jika file tidak ditemukan.
            Exception: Untuk error lainnya saat memuat file.
        """
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_path} tidak ditemukan.")
        except Exception as e:
            raise Exception(f"Error saat memuat file {file_path}: {str(e)}")

    def clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Membersihkan nama kolom dan menghapus kolom yang tidak dibutuhkan.

        Args:
            df (pd.DataFrame): DataFrame input.

        Returns:
            pd.DataFrame: DataFrame yang telah dibersihkan.
        """
        df.columns = df.columns.str.strip()
        logger.info("Sebelum menghapus kolom 'Patient Id':")
        logger.info(f"Kolom yang ada: {list(df.columns)}")
        logger.info("Data (5 baris pertama):\n%s", df.head().to_string())
        df = df.drop(columns=["index", "Patient Id"], errors="ignore")
        logger.info("Setelah menghapus kolom 'Patient Id':")
        logger.info(f"Kolom yang ada: {list(df.columns)}")
        logger.info("Data (5 baris pertama):\n%s", df.head().to_string())
        return df

    def encode_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Melakukan label encoding pada kolom target jika ada.

        Args:
            df (pd.DataFrame): DataFrame input.

        Returns:
            Tuple[pd.DataFrame, bool]: DataFrame yang telah diencode dan status keberhasilan.
        """
        target_candidates = ["Level", "Target", "Label", "Class"]
        target_column = next((col for col in target_candidates if col in df.columns), None)
        if target_column is None:
            logger.warning("Tidak ada kolom target ditemukan (kandidat: %s).", target_candidates)
            return df, False
        df[target_column] = LabelEncoder().fit_transform(df[target_column])
        logger.info("Kolom target '%s' berhasil diencode.", target_column)
        return df, True

    def encode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Melakukan one-hot encoding pada fitur kategorikal.

        Args:
            X (pd.DataFrame): DataFrame fitur.

        Returns:
            pd.DataFrame: DataFrame dengan fitur yang telah diencode.
        """
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) == 0:
            return X
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
        encoded_array = encoder.fit_transform(X[categorical_cols])
        encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
        X = X.drop(columns=categorical_cols).reset_index(drop=True)
        X = pd.concat([X, encoded_df], axis=1)
        return X

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Membagi data latih dan uji dengan rasio 80:20.

        Args:
            X (pd.DataFrame): Fitur.
            y (pd.Series): Target.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Data latih dan uji.
        """
        return train_test_split(X, y, test_size=0.2, random_state=42)


class FeatureSelector:
    def __init__(self):
        self.rfecv: Optional[RFECV] = None
        self.optimal_features: Optional[List[str]] = None
        self.rfe: Optional[RFE] = None
        self.rfe_features: Optional[List[str]] = None

    def run_rfecv(self, X_train: pd.DataFrame, y_train: pd.Series, best_params: Dict,
                  output_dir: str = "output") -> RFECV:
        """Melakukan seleksi fitur menggunakan RFECV berbasis Random Forest dengan parameter hasil tuning.

        Args:
            X_train (pd.DataFrame): Data latih fitur.
            y_train (pd.Series): Data latih target.
            best_params (Dict): Parameter terbaik hasil dari hyperparameter tuning.
            output_dir (str): Direktori untuk menyimpan output.

        Returns:
            RFECV: Objek RFECV yang telah dilatih.
        """
        os.makedirs(output_dir, exist_ok=True)
        base_model = RandomForestClassifier(**best_params, random_state=42)
        self.rfecv = RFECV(estimator=base_model, step=1, cv=5, scoring="accuracy", n_jobs=-1)
        self.rfecv.fit(X_train, y_train)
        self.optimal_features = X_train.columns[self.rfecv.support_].tolist()

        results = pd.DataFrame(self.rfecv.cv_results_)
        logger.info("RFECV – Rangkuman Evaluasi per Jumlah Fitur:")
        for i in range(len(results)):
            mean_score = results["mean_test_score"][i]
            std_score = results["std_test_score"][i]
            split_scores = [results[f"split{j}_test_score"][i] for j in range(5)]
            logger.info("Jumlah Fitur: %d", i + 1)
            logger.info("  Mean Accuracy: %.4f", mean_score)
            logger.info("  Std Dev: %.4f", std_score)
            for j, score in enumerate(split_scores):
                logger.info("    Fold %d: %.4f", j + 1, score)

        results_out = results[[
            "mean_test_score", "std_test_score",
            "split0_test_score", "split1_test_score",
            "split2_test_score", "split3_test_score", "split4_test_score"
        ]].copy()
        results_out.insert(0, "Jumlah Fitur", range(1, len(results_out) + 1))
        results_out.to_csv(os.path.join(output_dir, "rfecv_fold_results.csv"), index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(results["mean_test_score"]) + 1), results["mean_test_score"], marker='o', linestyle='--',
                 color='blue')
        plt.fill_between(
            range(1, len(results["mean_test_score"]) + 1),
            results["mean_test_score"] - results["std_test_score"],
            results["mean_test_score"] + results["std_test_score"],
            alpha=0.2, color='blue'
        )
        plt.xlabel("Jumlah Fitur yang Dipakai")
        plt.ylabel("Mean CV Accuracy")
        plt.title("RFECV: Akurasi vs Jumlah Fitur")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "rfecv_accuracy_curve.png"), dpi=300, bbox_inches="tight")
        plt.close()
        return self.rfecv

    def get_optimal_features(self) -> List[str]:
        """Mengembalikan daftar fitur terbaik hasil seleksi RFECV.

        Returns:
            List[str]: Daftar nama fitur optimal.

        Raises:
            ValueError: Jika run_rfecv belum dijalankan.
        """
        if self.optimal_features is None:
            raise ValueError("Jalankan run_rfecv terlebih dahulu.")
        return self.optimal_features

    def run_rfe(self, X_train: pd.DataFrame, y_train: pd.Series, n_features: int) -> RFE:
        """Melakukan seleksi fitur menggunakan RFE (tanpa CV).

        Args:
            X_train (pd.DataFrame): Data latih fitur.
            y_train (pd.Series): Data latih target.
            n_features (int): Jumlah fitur yang akan dipilih.

        Returns:
            RFE: Objek RFE yang telah dilatih.
        """
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rfe = RFE(estimator=base_model, n_features_to_select=n_features)
        self.rfe.fit(X_train, y_train)
        self.rfe_features = X_train.columns[self.rfe.support_].tolist()
        return self.rfe

    def get_rfe_features(self) -> List[str]:
        """Mengembalikan daftar fitur terbaik hasil seleksi RFE.

        Returns:
            List[str]: Daftar nama fitur optimal.

        Raises:
            ValueError: Jika run_rfe belum dijalankan.
        """
        if self.rfe_features is None:
            raise ValueError("Jalankan run_rfe terlebih dahulu.")
        return self.rfe_features


class RandomForestModel:
    def __init__(self):
        self.best_model: Optional[RandomForestClassifier] = None
        self.best_params: Optional[Dict] = None
        self.feature_importances: Optional[pd.Series] = None

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Mencari parameter terbaik dengan RandomizedSearchCV.

        Args:
            X_train (pd.DataFrame): Data latih fitur.
            y_train (pd.Series): Data latih target.

        Returns:
            Dict: Parameter terbaik.
        """
        base_model = RandomForestClassifier(random_state=42)
        param_dist = {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [3, 5, 8, 10, 15, None],
            "min_samples_split": [2, 4, 6, 8],
            "min_samples_leaf": [1, 2, 4, 6],
            "max_features": ["sqrt", "log2", None],
            "max_samples": [0.6, 0.8, 1.0],
            "criterion": ["gini", "entropy"],
        }
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=10,
            cv=5,
            scoring="accuracy",
            random_state=42,
            n_jobs=-1,
        )
        random_search.fit(X_train, y_train)
        self.best_params = random_search.best_params_

        results = pd.DataFrame(random_search.cv_results_)
        logger.info("Hasil Evaluasi Tiap Kombinasi Parameter:")
        for i in range(len(results)):
            logger.info("Kombinasi ke-%d: %s", i + 1, results['params'][i])
            logger.info("  Mean accuracy: %.4f", results['mean_test_score'][i])
            logger.info("  Std: %.4f", results['std_test_score'][i])
            for fold in range(5):
                logger.info("    Fold %d: %.4f", fold + 1, results[f'split{fold}_test_score'][i])

        return self.best_params

    def train_best_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Melatih model dengan parameter terbaik hasil tuning.

        Args:
            X_train (pd.DataFrame): Data latih fitur.
            y_train (pd.Series): Data latih target.

        Returns:
            RandomForestClassifier: Model yang telah dilatih.

        Raises:
            ValueError: Jika hyperparameter_tuning belum dijalankan.
        """
        if self.best_params is None:
            raise ValueError("Jalankan hyperparameter_tuning terlebih dahulu.")
        self.best_model = RandomForestClassifier(
            **self.best_params,
            random_state=42,
        )
        self.best_model.fit(X_train, y_train)
        self.feature_importances = pd.Series(
            self.best_model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)
        return self.best_model

    def evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Tuple[
        float, float, str, np.ndarray]:
        """Mengevaluasi model menggunakan data latih dan uji.

        Args:
            X_train (pd.DataFrame): Data latih fitur.
            X_test (pd.DataFrame): Data uji fitur.
            y_train (pd.Series): Data latih target.
            y_test (pd.Series): Data uji target.

        Returns:
            Tuple[float, float, str, np.ndarray]: Akurasi latih, akurasi uji, laporan klasifikasi, skor CV.

        Raises:
            ValueError: Jika train_best_model belum dijalankan.
        """
        if self.best_model is None:
            raise ValueError("Jalankan train_best_model terlebih dahulu.")
        y_train_pred = self.best_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        y_test_pred = self.best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred)
        cv_scores = cross_val_score(self.best_model, X_test, y_test, cv=5, scoring="accuracy")
        return train_accuracy, test_accuracy, report, cv_scores

    def get_feature_importance(self) -> pd.Series:
        """Menampilkan nilai kepentingan tiap fitur.

        Returns:
            pd.Series: Series berisi kepentingan fitur.

        Raises:
            ValueError: Jika train_best_model belum dijalankan.
        """
        if self.feature_importances is None:
            raise ValueError("Jalankan train_best_model terlebih dahulu.")
        return self.feature_importances


class VisualizationHelper:
    def __init__(self, output_dir: str = "output"):
        """Inisialisasi VisualizationHelper dengan direktori output.

        Args:
            output_dir (str): Direktori untuk menyimpan file output.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_all_visualizations(
            self,
            feature_importances: pd.Series,
            cv_scores: np.ndarray,
            train_accuracy: float,
            test_accuracy: float,
            approach: str = ""
    ) -> None:
        """Menjalankan semua visualisasi untuk satu pendekatan.

        Args:
            feature_importances (pd.Series): Kepentingan fitur.
            cv_scores (np.ndarray): Skor cross-validation.
            train_accuracy (float): Akurasi data latih.
            test_accuracy (float): Akurasi data uji.
            approach (str): Nama pendekatan (opsional).
        """
        self.plot_feature_importance(feature_importances.index, feature_importances, approach)
        self.plot_cv_accuracy_folds(cv_scores, approach)
        self.plot_cv_score_distribution(cv_scores, approach)
        self.plot_accuracy_comparison(train_accuracy, test_accuracy, np.mean(cv_scores), approach)

    def plot_feature_importance(self, feature_names: List, importances: pd.Series, approach: str = "") -> None:
        """Menampilkan grafik horizontal feature importance.

        Args:
            feature_names (List): Nama fitur.
            importances (pd.Series): Nilai kepentingan fitur.
            approach (str): Nama pendekatan (opsional).
        """
        plt.figure(figsize=(10, 5))
        plt.barh(feature_names, importances, color="skyblue")
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.title(f"Feature Importance dari Random Forest ({approach})")
        plt.savefig(os.path.join(self.output_dir, f"feature_importance_{approach.replace(' ', '_')}.png"), dpi=300,
                    bbox_inches="tight")
        plt.close()

    def plot_cv_accuracy_folds(self, cv_scores: np.ndarray, approach: str = "") -> None:
        """Visualisasi akurasi tiap fold.

        Args:
            cv_scores (np.ndarray): Skor cross-validation.
            approach (str): Nama pendekatan (opsional).
        """
        y_labels = [f"Fold {i + 1}" for i in range(len(cv_scores))]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=y_labels, y=cv_scores, hue=y_labels, palette="Blues", legend=False)
        plt.ylim(0.0, 1.0)
        plt.xlabel("Fold")
        plt.ylabel("Akurasi")
        plt.title(f"Akurasi Cross-Validation pada Setiap Fold ({approach})")
        plt.savefig(os.path.join(self.output_dir, f"cv_accuracy_folds_{approach.replace(' ', '_')}.png"), dpi=300,
                    bbox_inches="tight")
        plt.close()

    def plot_cv_score_distribution(self, cv_scores: np.ndarray, approach: str = "") -> None:
        """Visualisasi distribusi skor cross-validation.

        Args:
            cv_scores (np.ndarray): Skor cross-validation.
            approach (str): Nama pendekatan (opsional).
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(cv_scores, kde=True, color="skyblue", bins=len(cv_scores))
        plt.xlabel("Accuracy")
        plt.title(f"Distribusi Skor CV ({approach})")
        plt.savefig(os.path.join(self.output_dir, f"cv_score_distribution_{approach.replace(' ', '_')}.png"), dpi=300,
                    bbox_inches="tight")
        plt.close()

    def plot_hyperparameter_heatmap(self, df_params: pd.DataFrame) -> None:
        """Menampilkan heatmap kombinasi hyperparameter.

        Args:
            df_params (pd.DataFrame): DataFrame berisi kombinasi hyperparameter.
        """
        df_counts = df_params.astype(str).apply(pd.Series.value_counts).fillna(0)
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_counts, annot=True, cmap="plasma", fmt=".0f")
        plt.xlabel("Nilai")
        plt.ylabel("Hyperparameter")
        plt.title("Peringkat Hyperparameter Optimal dalam RandomizedSearchCV")
        plt.savefig(os.path.join(self.output_dir, "hyperparameter_ranking_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_accuracy_comparison(self, train_accuracy: float, test_accuracy: float, mean_cv_accuracy: float,
                                 approach: str = "") -> None:
        """Menampilkan barplot untuk membandingkan akurasi pelatihan, pengujian, dan rata-rata CV.

        Args:
            train_accuracy (float): Akurasi data latih.
            test_accuracy (float): Akurasi data uji.
            mean_cv_accuracy (float): Rata-rata akurasi CV.
            approach (str): Nama pendekatan (opsional).
        """
        accuracies = [train_accuracy, test_accuracy, mean_cv_accuracy]
        labels = ['Training Accuracy', 'Testing Accuracy', 'Mean CV Accuracy']
        plt.figure(figsize=(8, 5))
        sns.barplot(x=labels, y=accuracies, hue=labels, palette="viridis", legend=False)
        plt.ylim(0, 1)
        plt.ylabel("Akurasi")
        plt.title(f"Perbandingan Akurasi Model ({approach})")
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f"{acc:.3f}", ha='center', va='bottom')
        plt.savefig(os.path.join(self.output_dir, f"accuracy_comparison_{approach.replace(' ', '_')}.png"), dpi=300,
                    bbox_inches="tight")
        plt.close()

    def plot_data_distribution(self, df: pd.DataFrame) -> None:
        """Visualisasi distribusi data jika tidak ada target.

        Args:
            df (pd.DataFrame): DataFrame input.
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            logger.warning("Tidak ada kolom numerik untuk visualisasi distribusi.")
            return
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(numeric_columns, 1):
            plt.subplot(2, (len(numeric_columns) + 1) // 2, i)
            sns.histplot(df[col], kde=True, color="green")
            plt.title(f"Distribusi {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "data_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_correlation_matrix(self, df: pd.DataFrame, title_suffix: str = "") -> None:
        """Menampilkan heatmap matriks korelasi antar fitur dalam dataset.

        Args:
            df (pd.DataFrame): DataFrame yang berisi fitur untuk dihitung korelasi.
            title_suffix (str): Sufiks untuk judul plot (opsional).
        """
        correlation_matrix = df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
        plt.title(f"Matriks Korelasi Antar Fitur {title_suffix}")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"correlation_matrix_{title_suffix.replace(' ', '_')}.png"), dpi=300,
                    bbox_inches="tight")
        plt.close()


def run_approach(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        approach_name: str,
        model: RandomForestModel,
        visualizer: VisualizationHelper
) -> Tuple[float, float, np.ndarray, pd.Series]:
    """Menjalankan pelatihan, evaluasi, dan visualisasi untuk satu pendekatan.

    Args:
        X_train (pd.DataFrame): Data latih fitur.
        X_test (pd.DataFrame): Data uji fitur.
        y_train (pd.Series): Data latih target.
        y_test (pd.Series): Data uji target.
        approach_name (str): Nama pendekatan.
        model (RandomForestModel): Objek model Random Forest.
        visualizer (VisualizationHelper): Objek visualisasi.

    Returns:
        Tuple[float, float, np.ndarray, pd.Series]: Akurasi latih, akurasi uji, skor CV, kepentingan fitur.
    """
    logger.info("Memproses Pendekatan: %s", approach_name)
    best_params = model.hyperparameter_tuning(X_train, y_train)
    logger.info("Best Parameters (%s): %s", approach_name, best_params)
    model.train_best_model(X_train, y_train)
    train_acc, test_acc, report, cv_scores = model.evaluate(X_train, X_test, y_train, y_test)
    logger.info("Cross-Validation Accuracy Scores (%s): %s", approach_name, cv_scores)
    logger.info("Mean CV Accuracy (%s): %.4f", approach_name, np.mean(cv_scores))
    logger.info("Training Accuracy (%s): %.4f", approach_name, train_acc)
    logger.info("Testing Accuracy (%s): %.4f", approach_name, test_acc)
    logger.info("Classification Report (%s):\n%s", approach_name, report)
    feature_importances = model.get_feature_importance()
    logger.info("Top Feature Importance (%s):", approach_name)
    logger.info("\n%s", feature_importances.head(10).to_string())
    visualizer.plot_all_visualizations(feature_importances, cv_scores, train_acc, test_acc, approach_name)
    return train_acc, test_acc, cv_scores, feature_importances


if __name__ == "__main__":
    preprocessor = PreprocessingHandler()
    feature_selector = FeatureSelector()
    visualizer = VisualizationHelper(output_dir="output")

    try:
        # Load dan preprocess data
        file_path = "cancer patient data sets.csv"
        df = preprocessor.load_dataset(file_path)
        df = preprocessor.clean_columns(df)

        logger.info("Tabel 3.1: Summary Statistics of Data")
        summary_stats = preprocessor.generate_summary_statistics(df)
        logger.info("\n%s", summary_stats.to_string(index=False))

        df, target_encoded = preprocessor.encode_target(df)

        if target_encoded:
            target_column = next(col for col in ["Level", "Target", "Label", "Class"] if col in df.columns)
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X = preprocessor.encode_features(X)
            X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

            visualizer.plot_correlation_matrix(X, title_suffix="All Features")
            metrics = []

            # Pendekatan 1: Semua fitur
            model_all = RandomForestModel()
            train_acc_all, test_acc_all, cv_scores_all, feature_importances_all = run_approach(
                X_train, X_test, y_train, y_test, "All Features", model_all, visualizer
            )
            metrics.append({"Approach": "All Features", "Training Accuracy": train_acc_all,
                            "Testing Accuracy": test_acc_all, "Mean CV Accuracy": np.mean(cv_scores_all)})

            # Hyperparameter tuning khusus untuk RFECV
            model_rfecv_tuning = RandomForestModel()
            best_params_rfecv = model_rfecv_tuning.hyperparameter_tuning(X_train, y_train)
            logger.info("Best Parameters for RFECV: %s", best_params_rfecv)

            # RFECV dengan best_params hasil tuning
            feature_selector.run_rfecv(X_train, y_train, best_params=best_params_rfecv, output_dir="output")
            rfecv_features = feature_selector.get_optimal_features()
            logger.info("Optimal Features (RFECV): %s", rfecv_features)
            logger.info("Jumlah Fitur Optimal (RFECV): %d", len(rfecv_features))
            X_train_rfecv = X_train[rfecv_features]
            X_test_rfecv = X_test[rfecv_features]
            visualizer.plot_correlation_matrix(X_train_rfecv, title_suffix="RFECV Selected")

            # Evaluasi RFECV seperti biasa
            model_rfecv = RandomForestModel()
            train_acc_rfecv, test_acc_rfecv, cv_scores_rfecv, feature_importances_rfecv = run_approach(
                X_train_rfecv, X_test_rfecv, y_train, y_test, "RFECV", model_rfecv, visualizer
            )
            metrics.append({"Approach": "RFECV Selected", "Training Accuracy": train_acc_rfecv,
                            "Testing Accuracy": test_acc_rfecv, "Mean CV Accuracy": np.mean(cv_scores_rfecv)})

            top_feats_by_importance = feature_importances_all.index[:len(rfecv_features)].tolist()
            common_feats = set(rfecv_features).intersection(top_feats_by_importance)
            logger.info("Fitur RFECV yang juga termasuk dalam top-%d feature importance: %s", len(rfecv_features),
                        list(common_feats))
            logger.info("Fitur RFECV unik (tidak dalam top-%d importance): %s", len(rfecv_features),
                        [f for f in rfecv_features if f not in common_feats])
            logger.info("Fitur penting unik (dalam top-%d importance tapi tidak dipilih RFECV): %s",
                        len(rfecv_features), [f for f in top_feats_by_importance if f not in common_feats])

            # Pendekatan 3: RFE
            n_rfe = len(rfecv_features)
            feature_selector.run_rfe(X_train, y_train, n_rfe)
            rfe_features = feature_selector.get_rfe_features()
            logger.info("Optimal Features (RFE): %s", rfe_features)
            logger.info("Jumlah Fitur RFE: %d", len(rfe_features))
            X_train_rfe = X_train[rfe_features]
            X_test_rfe = X_test[rfe_features]

            model_rfe = RandomForestModel()
            train_acc_rfe, test_acc_rfe, cv_scores_rfe, feature_importances_rfe = run_approach(
                X_train_rfe, X_test_rfe, y_train, y_test, "RFE", model_rfe, visualizer
            )
            metrics.append({"Approach": "RFE Selected", "Training Accuracy": train_acc_rfe,
                            "Testing Accuracy": test_acc_rfe, "Mean CV Accuracy": np.mean(cv_scores_rfe)})

            # Pendekatan 4: No Top 5 Features
            top_features_to_remove = feature_importances_all.index[:5]
            logger.info("Menghapus fitur paling berpengaruh: %s", list(top_features_to_remove))
            X_train_no_top = X_train.drop(columns=top_features_to_remove)
            X_test_no_top = X_test.drop(columns=top_features_to_remove)

            model_no_top = RandomForestModel()
            train_acc_no_top, test_acc_no_top, cv_scores_no_top, feature_importances_no_top = run_approach(
                X_train_no_top, X_test_no_top, y_train, y_test, "No Top 5", model_no_top, visualizer
            )
            metrics.append({"Approach": "No Top 5 Features", "Training Accuracy": train_acc_no_top,
                            "Testing Accuracy": test_acc_no_top, "Mean CV Accuracy": np.mean(cv_scores_no_top)})

            # Perbandingan metrik akurasi
            metrics_df = pd.DataFrame(metrics)
            logger.info("Perbandingan Metrik Akurasi:")
            logger.info("\n%s", metrics_df.to_string(index=False))

            # Grafik perbandingan akurasi gabungan
            acc_data = pd.DataFrame({
                "Approach": np.repeat(["All Features", "RFECV", "RFE", "No Top 5"], 3),
                "Metric": ["Training Accuracy", "Testing Accuracy", "Mean CV Accuracy"] * 4,
                "Accuracy": [train_acc_all, test_acc_all, np.mean(cv_scores_all),
                             train_acc_rfecv, test_acc_rfecv, np.mean(cv_scores_rfecv),
                             train_acc_rfe, test_acc_rfe, np.mean(cv_scores_rfe),
                             train_acc_no_top, test_acc_no_top, np.mean(cv_scores_no_top)]
            })
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Approach", y="Accuracy", hue="Metric", data=acc_data, palette="pastel")
            plt.ylim(0, 1)
            plt.ylabel("Akurasi")
            plt.title("Perbandingan Akurasi Antar Pendekatan")
            plt.legend(loc='lower right')
            plt.savefig(os.path.join("output", "accuracy_comparison_combined.png"), dpi=300, bbox_inches="tight")
            plt.close()

            # Visualisasi hyperparameter
            df_params = pd.DataFrame([
                model_all.best_params,
                model_rfecv.best_params,
                model_rfe.best_params,
                model_no_top.best_params
            ])
            visualizer.plot_hyperparameter_heatmap(df_params)

        else:
            logger.info("Melanjutkan dengan visualisasi distribusi data karena target tidak ditemukan.")
            X = preprocessor.encode_features(df)
            visualizer.plot_data_distribution(X)

    except Exception as e:
        logger.error("Terjadi kesalahan: %s", str(e))
        raise