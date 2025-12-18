from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)


def build_feature_pipeline():

    stages = []

    # 1. Mã hóa location (categorical)
    loc_indexer = StringIndexer(
        inputCol="location",
        outputCol="location_index",
        handleInvalid="keep"
    )

    loc_encoder = OneHotEncoder(
        inputCol="location_index",
        outputCol="location_encoded"
    )

    # 2. Mã hóa season (categorical - thêm từ EDA)
    season_indexer = StringIndexer(
        inputCol="season",
        outputCol="season_index",
        handleInvalid="keep"
    )

    season_encoder = OneHotEncoder(
        inputCol="season_index",
        outputCol="season_encoded"
    )

    # 3. Chuẩn hóa numerical features
    numeric_cols = ["temperature_c", "humidity_pct",
                    "wind_speed_kmh", "pressure_hpa"]

    # Tạo VectorAssembler cho numeric features
    numeric_assembler = VectorAssembler(
        inputCols=numeric_cols,
        outputCol="numeric_features"
    )

    # StandardScaler
    scaler = StandardScaler(
        inputCol="numeric_features",
        outputCol="scaled_numeric_features",
        withStd=True,
        withMean=True
    )

    # 4. Kết hợp tất cả features
    assembler = VectorAssembler(
        inputCols=[
            "location_encoded",
            "season_encoded",
            "scaled_numeric_features"
        ],
        outputCol="features"
    )

    stages += [
        loc_indexer,
        loc_encoder,
        season_indexer,
        season_encoder,
        numeric_assembler,
        scaler,
        assembler
    ]

    return stages


