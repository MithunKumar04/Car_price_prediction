import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, lower, regexp_replace, col,when
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("Car Price Prediction").getOrCreate()

# Load data
data_path = "CarPrice_Assignment.csv"  # Update path as needed
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Preprocess data
df = df.withColumn("CarName", lower(split(col("CarName"), " ").getItem(0)))
df = df.withColumn(
    "CarName",
    when(col("CarName") == "maxda", "mazda")
    .when(col("CarName") == "porcshce", "porsche")
    .when(col("CarName") == "toyouta", "toyota")
    .when(col("CarName") == "vokswagen", "volkswagen")
    .when(col("CarName") == "vw", "volkswagen")
    .otherwise(col("CarName"))
)
df = df.drop("symboling", "car_ID")

# String Indexing for categorical variables
categorical_cols = [col for col, dtype in df.dtypes if dtype == "string"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index").fit(df) for col in categorical_cols]
for indexer in indexers:
    df = indexer.transform(df)

# Define features and target
feature_cols = [col for col in df.columns if col not in ["price"] + categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Train-test split
train_data, test_data = df.randomSplit([0.75, 0.25], seed=42)

# Train Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="price")
lr_model = lr.fit(train_data)

# Train Random Forest Regressor model
rf = RandomForestRegressor(featuresCol="features", labelCol="price")
rf_model = rf.fit(train_data)

# Streamlit UI
st.title("Car Price Prediction App")
st.write("Enter the details below to predict the car price.")

# Create user input fields for prediction
car_data = {}
for column in feature_cols:
    if column.endswith("_index"):  # Categorical feature
        unique_vals = df.select(column).distinct().rdd.flatMap(lambda x: x).collect()
        car_data[column] = st.selectbox(f"{column.replace('_index', '')}", unique_vals)
    else:  # Numerical feature
        col_min = df.select(col(column)).rdd.map(lambda x: x[0]).min()
        col_max = df.select(col(column)).rdd.map(lambda x: x[0]).max()
        car_data[column] = st.number_input(f"{column}", min_value=float(col_min), max_value=float(col_max), step=0.1)

# Convert input to DataFrame for prediction
input_df = spark.createDataFrame([car_data])

# Assemble features for prediction
input_df = assembler.transform(input_df)

# Make predictions
if st.button("Predict Price"):
    pred_lr = lr_model.transform(input_df).select("prediction").collect()[0][0]
    pred_rf = rf_model.transform(input_df).select("prediction").collect()[0][0]

    # Display results
    st.write(f"*Linear Regression Prediction:* ₹{pred_lr:.2f}")
    st.write(f"*Random Forest Prediction:* ₹{pred_rf:.2f}")

st.write("*Note:* These models were trained on sample data and may need more training or tuning for production use.")
