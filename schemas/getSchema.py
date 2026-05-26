import duckdb

con = duckdb.connect()

schema = con.execute("""
DESCRIBE
SELECT *
FROM '../data/raw/parquet/train/*.parquet'; -- Ajuste o caminho para um dos arquivos Parquet gerados
""").fetchdf()

schema.to_csv("schema.csv", index=False)