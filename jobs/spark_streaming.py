import time
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import from_json, col, when, udf
from config.config import config
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, ChatResponse


llm = Ollama(model="llama3", request_timeout=60.0, base_url="http://ollama:11434")


def sentiment_analysis(comment: str) -> str:
    if comment:
        messages = [
            ChatMessage(
                role="system", content="""
                    You're a machine learning model with a task of classifying comments into POSITIVE, NEGATIVE. NEUTRAL.
                    You are to respond with one word from the option specified above, do not add anything else.
                """
            ),
            ChatMessage(role="user", content=comment),
        ]
    resp: ChatResponse = llm.chat(messages)
    return resp.message.content


def start_streaming(spark: SparkSession):
    topic = "customers_reviews"
    try:
        stream_df = (
            spark.readStream
            .format("socket")
            .option("host", "0.0.0.0")
            .option("port", 9999)
            .load()
        )

        schema = StructType([
            StructField("review_id", StringType()),
            StructField("user_id", StringType()),
            StructField("business_id", StringType()),
            StructField("stars", FloatType()),
            # StructField("cool", FloatType()),
            # StructField("funny", FloatType()),
            # StructField("useful", FloatType()),
            StructField("date", StringType()),
            StructField("text", StringType())
        ])

        stream_df = stream_df.select(from_json(col("value"), schema).alias("data")).select(("data.*"))
        stream_df.printSchema()

        sentiment_analysis_udf = udf(sentiment_analysis, StringType())

        stream_df = stream_df.withColumn(
            "feedback",
            when(col("text").isNotNull(), sentiment_analysis_udf(col("text")))
            .otherwise(None)
        )

        kafka_df = stream_df.selectExpr("CAST(review_id AS STRING) AS key", "to_json(struct(*)) AS value")

        query = (
            kafka_df.writeStream
            .format("kafka")
            .option("kafka.bootstrap.servers", config.get("kafka").get("bootstrap.servers"))
            .option("kafka.security.protocol", config.get("kafka").get("security.protocol"))
            .option("kafka.sasl.mechanism", config.get("kafka").get("sasl.mechanism"))
            .option('kafka.sasl.jaas.config',
                    'org.apache.kafka.common.security.plain.PlainLoginModule required username="{username}" '
                    'password="{password}";'.format(
                        username=config['kafka']['sasl.username'],
                        password=config['kafka']['sasl.password']
                    ))
            .option("checkpointLocation", "/tmp/checkpoint")
            .option("topic", topic)
            .start()
            .awaitTermination()
        )

        # query = stream_df.writeStream.outputMode("append").format("console").options(truncate=True).start()
        # query.awaitTermination()
    except Exception as e:
        print(f"Exception encountered: {e}. Retrying in 10s")
        time.sleep(10)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("SocketStreamConsumer").getOrCreate()

    start_streaming(spark)