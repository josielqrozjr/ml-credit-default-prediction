from pathlib import Path
import duckdb
import time

# =========================
# CONFIGURAÇÕES
# =========================

RAW_DIR = Path("data/raw/csv")

TRAIN_LABELS_CSV = RAW_DIR / "train_labels.csv"
TRAIN_CSV = RAW_DIR / "train_data.csv"

TRAIN_LABELS_OUTPUT = Path("data/raw/parquet/train_labels")
TRAIN_OUTPUT = Path("data/raw/parquet/train")

TRAIN_LABELS_OUTPUT.mkdir(parents=True, exist_ok=True)
TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)

# Quantidade de linhas por row group
# (ajuste conforme RAM)
ROW_GROUP_SIZE = 250_000

# =========================
# CONEXÃO DUCKDB
# =========================

con = duckdb.connect(database=":memory:")

# Menos threads = menos RAM
con.execute("PRAGMA threads=2")

# Limite de RAM
con.execute("PRAGMA memory_limit='8GB'")

# Diretório temporário
con.execute("PRAGMA temp_directory='tmp_duckdb'")

# =========================
# FUNÇÃO DE CONVERSÃO
# =========================

def convert_csv_to_parquet(csv_path, output_dir, dataset_name):

    print(f"\nIniciando conversão: {dataset_name}")
    start = time.time()

    query = f"""
COPY (
    SELECT *
    FROM read_csv(
        '{csv_path}',
        auto_detect=true,
        header=true
    )
)
TO '{output_dir}'
(
    FORMAT PARQUET,
    COMPRESSION SNAPPY,
    ROW_GROUP_SIZE {ROW_GROUP_SIZE},
    PER_THREAD_OUTPUT,
    OVERWRITE_OR_IGNORE
);
    """

    con.execute(query)

    elapsed = time.time() - start

    print(f"{dataset_name} convertido com sucesso!")
    print(f"Tempo: {elapsed/60:.2f} minutos")


# =========================
# EXECUÇÃO
# =========================

# convert_csv_to_parquet(
#     TRAIN_LABELS_CSV,
#     TRAIN_LABELS_OUTPUT,
#     "TRAIN_LABELS"
# )

convert_csv_to_parquet(
    TRAIN_CSV,
    TRAIN_OUTPUT,
    "TRAIN"
)

print("\nConversão finalizada.")
