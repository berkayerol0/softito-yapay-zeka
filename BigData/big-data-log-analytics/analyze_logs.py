from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, sum, max, min, window, desc, hour, when
import time

spark = SparkSession.builder \
    .appName("WebLogAnalysis") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("\n=== 1. Veriyi Yükleme ===")
start = time.time()
df = spark.read.option("header", "true").csv("logs/*.csv")
df = df.withColumn("timestamp", col("timestamp").cast("bigint")) \
       .withColumn("response_time_ms", col("response_time_ms").cast("int")) \
       .withColumn("bytes_sent", col("bytes_sent").cast("int")) \
       .withColumn("status", col("status").cast("int"))
total = df.count()
print(f"Yüklenen satir: {total:,} | Sure: {time.time()-start:.2f}s")

print("\n=== 2. Şema ===")
df.printSchema()

print("\n=== 3. HTTP Durum Kodları Dağılımı ===")
start = time.time()
df.groupBy("status").count().orderBy("status").show()
print(f"Süre: {time.time()-start:.2f}s")

print("\n=== 4. En Çok İstek Alan URL'ler (Top 10) ===")
start = time.time()
df.groupBy("url").count().orderBy(desc("count")).show(10)
print(f"Süre: {time.time()-start:.2f}s")

print("\n=== 5. Ülkelere Göre İstek Sayısı ve Ortalama Yanıt Süresi ===")
start = time.time()
df.groupBy("country").agg(
    count("*").alias("request_count"),
    avg("response_time_ms").alias("avg_response_ms"),
    avg("bytes_sent").alias("avg_bytes_sent")
).orderBy(desc("request_count")).show()
print(f"Süre: {time.time()-start:.2f}s")

print("\n=== 6. Saatlik İstek Dağılımı ===")
start = time.time()
df.withColumn("hour", hour((col("timestamp").cast("timestamp")))) \
  .groupBy("hour").count().orderBy("hour").show(24)
print(f"Süre: {time.time()-start:.2f}s")

print("\n=== 7. Yavaş URL'ler (ortalama > 2000ms, Top 10) ===")
start = time.time()
df.groupBy("url").agg(
    avg("response_time_ms").alias("avg_response"),
    count("*").alias("count")
).filter(col("avg_response") > 2000).orderBy(desc("avg_response")).show(10)
print(f"Süre: {time.time()-start:.2f}s")

print("\n=== 8. HTTP Hata Oranları (4xx/5xx) ===")
start = time.time()
df.withColumn("is_error", when(col("status") >= 400, 1).otherwise(0)) \
  .groupBy("url").agg(
      count("*").alias("total"),
      sum("is_error").alias("errors")
  ).withColumn("error_rate_pct", (col("errors") / col("total") * 100)) \
   .orderBy(desc("error_rate_pct")).show(10)
print(f"Süre: {time.time()-start:.2f}s")

print("\n=== 9. En Aktif IP'ler (Top 10) ===")
start = time.time()
df.groupBy("ip").count().orderBy(desc("count")).show(10)
print(f"Süre: {time.time()-start:.2f}s")

print("\n=== 10. Partition Sayısı & Veri Boyutu ===")
total_bytes = df.select(sum("bytes_sent")).collect()[0][0]
print(f"Toplam bayt gönderimi: {total_bytes:,}")
print(f"Partition sayısı: {df.rdd.getNumPartitions()}")

spark.stop()
print("\n✅ Analiz tamamlandı!")
