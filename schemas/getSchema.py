import duckdb

con = duckdb.connect()

schema = con.execute("""
DESCRIBE
SELECT *
FROM 'data/raw/parquet/train/*0.parquet';
""").fetchdf()

schema.to_csv("schemas/schema.csv", index=False)