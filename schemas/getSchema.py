import duckdb

con = duckdb.connect()

schema = con.execute("""
DESCRIBE
SELECT *
FROM '../data/raw/parquet/train/*.parquet'; -- Ajuste o caminho para um dos arquivos Parquet gerados
""").fetchdf()

schema.to_csv("schema.csv", index=False)

"""
CATEGORICAL_FEATURES = [
    'B_30', 'B_38', 'D_114', 'D_116',
    'D_117', 'D_120', 'D_126',
    'D_63', 'D_64', 'D_66', 'D_68'
]
"""