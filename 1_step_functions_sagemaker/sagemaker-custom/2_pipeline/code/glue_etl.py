import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

#Retrieve parameters for the Glue job.
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_SOURCE', 'S3_DEST',
                        'TRAIN_KEY', 'VAL_KEY'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

#Create a PySpark dataframe from the source table.
source_data_frame = spark.read.load(args['S3_SOURCE'], format='csv',
                                    inferSchema=True, header=True)

# Drop unused columns
columns_to_drop = ['Phone', 'Day Charge', 'Eve Charge', 'Night Charge', 'Intl Charge']
source_data_frame = source_data_frame.drop(*columns_to_drop)

# Change data type of 'Area Code' columns to string
# source_data_frame['Area Code'] = source_data_frame['Area Code'].astype(object) # in Pandas
source_data_frame = source_data_frame.withColumn('Area Code', source_data_frame['Area Code'].cast('string'))

print("source_data_frame before get_dummies:", source_data_frame.columns)

import pyspark.sql.functions as F 
# model_data = pd.get_dummies(source_data_frame) # in Pandas
def get_col_dummies(df, col_name):
    categ = df.select(col_name).distinct().rdd.flatMap(lambda x:x).collect()
    exprs = [F.when(F.col(col_name) == cat,1).otherwise(0)\
                .alias('`'+col_name+'_'+str(cat)+'`') for cat in categ]
    df = df.select(exprs+df.columns)
    print('Columns before dropping the original one in get dummies:', df.columns)
    df = df.drop(col_name)
    return df

categorical_cols = ['Area Code', 'Churn?', "Int'l Plan", 'State', 'VMail Plan']
for cat_col in categorical_cols:
    print('Creating dummy for column', cat_col)
    source_data_frame = get_col_dummies(source_data_frame, cat_col)

print("source_data_frame after get_dummies:", source_data_frame.columns)

source_data_frame = source_data_frame.drop('Churn?_False.')

cols = source_data_frame.columns
y = cols.pop()
cols.insert(0,y)

# Reorder columns putting target variable ?Churn as first column
source_data_frame = source_data_frame.select(cols)

#Split the dataframe in to training and validation dataframes.
train_data, val_data = source_data_frame.randomSplit([.7,.3])

#Write both dataframes to the destination datastore.
train_path = args['S3_DEST'] + args['TRAIN_KEY']
val_path = args['S3_DEST'] + args['VAL_KEY']

train_data.write.save(train_path, format='csv', mode='overwrite', header=True)
val_data.write.save(val_path, format='csv', mode='overwrite', header=True)

#Complete the job.
job.commit()
