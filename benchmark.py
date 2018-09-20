"""
Simple benchmarking of Jaro-Winkler string distance applied to data in PySpark.

The scala-udf-benchmark*.jar contains a Scala wrapped call to the Apache Commons implementation of Jaro-Winkler.
See :setup_spark_env: to see have that is registered with the SparkSession.

The Python implementation is taken from the jellyfish library.

This script saves it's results to csv, png in the results directory.

NOTE:
     This is designed to be run within the VM set up by Vagrant.
     It may take some tweaking to run elsewhere.
"""
import datetime
import itertools
import logging
import random
import timeit

import jellyfish
import nltk
import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as ps_funcs
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

__DATASET_MULTIPLIER_FACTOR = 4 * 1000
__DATASET_PARTITIONS = 2

__TRIAL_REPEATS = 10


def _setup_logging():
    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(f"logs/benchmark-{datetime.datetime.now().strftime('%Y-%m-%dT%H%M')}.log")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = _setup_logging()


def _fetch_random_words(n=1000):
    """Generate a random list of words"""

    # Ensure the same words each run
    random.seed(42)

    # Download the corpus if not present
    nltk.download('words')

    word_list = nltk.corpus.words.words()

    random.shuffle(word_list)

    random_words = word_list[:n]

    return random_words


def _random_phrase_generator(n=1000, phrase_length=3):
    """Generate 'phrases' of random words, i.e. take random words and join with spaces"""

    random_words = _fetch_random_words(n * phrase_length)

    keyed_random_words = zip(itertools.cycle(range(n)), random_words)

    sorted_keyed_random_words = sorted(keyed_random_words, key=lambda x: x[0])

    for key, group in itertools.groupby(sorted_keyed_random_words, key=lambda x: x[0]):
        yield ' '.join(word for k, word in group)


def _fetch_phrase_pairs() -> pd.DataFrame:
    """Returns a pd.DataFrame with pairs of random phrases."""

    phrases = list(_random_phrase_generator())

    word_pairs = list(zip(phrases, reversed(phrases)))

    df = pd.DataFrame(word_pairs)

    return df


def _setup_dataframe(spark, sqlContext, dataset_multiplier_factor, append_ids=True) -> pyspark.sql.DataFrame:
    """Setup a pyspark dataframe to run against.

    Then creates a PySpark dataframe, and crossjoins with a table of length :dataset_multiplier_factor:
    to increase the volume of data for benchmarking.

    Returns:
        A Pyspark dataframe with random phrases for string distance testing.
    """
    df = _fetch_phrase_pairs()

    logger.info(f'{len(df):,} word pairs')

    pyspark_df = spark.createDataFrame(df, ['left', 'right'])

    pyspark_df = pyspark_df.repartition(10)
    pyspark_df.cache().count()

    logger.debug('Increasing data volume')

    range_df = sqlContext.range(dataset_multiplier_factor)

    if append_ids:

        range_df = range_df.withColumn('id_string', ps_funcs.lpad('id', 12, "0"))

        pyspark_df = range_df.crossJoin(pyspark_df).select(
            ps_funcs.concat_ws(' ', ps_funcs.col('left'), ps_funcs.col('id_string')).alias('left'),
            ps_funcs.concat_ws(' ', ps_funcs.col('right'), ps_funcs.col('id_string')).alias('right')
        )
    else:
        pyspark_df = range_df.crossJoin(pyspark_df).select(
            ps_funcs.col('left'),
            ps_funcs.col('right')
        )

    pyspark_df = pyspark_df.repartition(__DATASET_PARTITIONS)
    record_count = pyspark_df.cache().count()

    logger.info(f'Generated dataframe with {record_count:,} records')

    sample_data = pyspark_df.sample(withReplacement=False, fraction=0.01).limit(1).collect()
    logger.info(f'Sample of benchmarking data: {sample_data}')

    return pyspark_df


def _run_trials(statement='pass'):

    durations = []
    for i in range(__TRIAL_REPEATS):
        logger.info(f'Running statement={statement}, trial={i}')
        duration = timeit.timeit(statement, globals=globals(), number=1)
        durations.append(duration)

    return durations


def _run_benchmarks(spark, sqlContext, dataset_multiplier_factor=__DATASET_MULTIPLIER_FACTOR, append_ids=True):

    spark.catalog.clearCache()

    global df
    df = _setup_dataframe(spark, sqlContext, dataset_multiplier_factor, append_ids)

    scala_results = _run_trials("scala_udf_distance(df)")
    logger.info(f'scala benchmark results: {scala_results}')
    logger.info(f'Median: {np.median(scala_results)}')

    python_results = _run_trials("python_udf_distance(df)")
    logger.info(f'python benchmark results: {python_results}')
    logger.info(f'Median: {np.median(python_results)}')

    record_count = df.count()

    results = []
    for result in scala_results:
        results.append({'implementation': 'scala', 'records': record_count, 'run_time': result})

    for result in python_results:
        results.append({'implementation': 'python', 'records': record_count, 'run_time': result})

    return results


def _plot_results(result_df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    def plot_run_time(result_df):
        fig = sns.lineplot(data=result_df, x='records', y='run_time', hue='implementation', markers=True, dashes=False,
                     style='implementation').get_figure()
        fig.savefig('results/run_time_plot.png', dpi=300)
        plt.close()

    plot_run_time(result_df)

    def plot_perf_ratio(result_df):
        medians_df = result_df.groupby(['records', 'implementation'])['run_time'].median().reset_index().pivot(index='records',
                columns='implementation').reset_index()

        medians_df['performance_ratio'] = medians_df[('run_time', 'scala')] / medians_df[('run_time', 'python')]
        medians_df['dummy'] = 'dummy'

        fig = sns.lineplot(data=medians_df, x='records', y='performance_ratio', hue='dummy', style='dummy', legend=False,
                     markers=True, ).get_figure()
        fig.savefig('results/median_performance_ratios.png', dpi=300)
        plt.close()

    plot_perf_ratio(result_df)


def setup_spark_env():
    """Create a standalone SparkSession.

    Also registers the JaroWinklerDistance function.
    """

    logger.info('Setting up spark env')

    # Important to ensure enough memory for the executor to process each partition

    spark = (SparkSession.builder.master('local[*]')
             .config('spark.executor.memory', '2g')
             .config('spark.executor.memoryOverhead', '1g')
             .config('spark.driver.memory', '2g')
             .config('spark.driver.memoryOverhead', '1g')
             .config('spark.python.worker.memory', '300m')
             .config('spark.default.parallelism', 200)
             .config('spark.driver.extraClassPath', '/vagrant/resources/scala-udf-benchmark-0.0.1-SNAPSHOT.jar')
             .config('spark.jars', '/vagrant/resources/scala-udf-benchmark-0.0.1-SNAPSHOT.jar').getOrCreate())

    sqlContext = SQLContext(spark.sparkContext)

    sqlContext.registerJavaFunction('jaro_winkler', 'uk.gov.ons.mdr.examples.JaroWinklerDistance', DoubleType())

    logger.debug('Check function to benchmark is callable')
    spark.sql("""SELECT jaro_winkler('ABC Corporation', 'ABC Corp')""").show()

    return spark, sqlContext


def non_spark_benchmark():

    global word_pair_pandas_df
    word_pair_pandas_df = _fetch_phrase_pairs()

    python_results = _run_trials("pandas_distance(word_pair_pandas_df)")

    logger.info(f'Non Spark python benchmark results: {python_results}')
    logger.info(f'Median: {np.median(python_results)} for {len(word_pair_pandas_df)} records')


def single_benchmark(spark, sqlContext, mult=__DATASET_MULTIPLIER_FACTOR, append_ids=True):
    logger.info('Running single dataset benchmark')
    result = _run_benchmarks(spark, sqlContext, mult, append_ids)

    result_df = pd.DataFrame(result)
    result_df.to_csv(f'results/single_benchmark_{mult}.csv', index=False)

    result_df = pd.DataFrame(result).groupby('implementation')['run_time'].median()

    logger.info(f'Bench mark results: {result_df}')
    print(result_df)
    return result


def multiple_benchmarks(spark, sqlContext):

    multipliers = list(range(100, 1001, 100))

    logger.info(f'Running multiplied dataset benchmark for factors: {multipliers}')

    results = []
    for mult in multipliers:
        logger.debug(f'Running for multiplier={mult}')
        result = _run_benchmarks(spark, sqlContext, mult)
        results += result

    result_df = pd.DataFrame(results)

    filename = f"results/benchmarking_results_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
    logger.info(f'Saving results to {filename}')
    result_df.to_csv(filename, index=False)

    _plot_results(result_df)
    return result_df


def pandas_distance(df):
    jaro_vect = np.vectorize(jellyfish.jaro_winkler)
    result = np.sum(jaro_vect(df[0], df[1]))
    logger.debug(f'Sum of distances: {result}')
    return result


def scala_udf_distance(df):
    """Calls the Scala UDF to benchmark"""
    result = df.selectExpr('jaro_winkler(left, right) as distance').select(ps_funcs.sum('distance')).collect()
    logger.info(f'Sum of distances: {result}')
    return result

@udf(DoubleType())
def jaro_winkler_python(left, right):
    """Python UDF to compute Jaro Winkler distance between strings"""
    return jellyfish.jaro_winkler(left, right)


def python_udf_distance(df):
    """Calls the Python UDF to benchmark"""
    result = df.select(jaro_winkler_python('left', 'right').alias('distance')).select(ps_funcs.sum('distance')).collect()
    logger.info(f'Sum of distances: {result}')
    return result


def main():

    # How fast is the Python function without Spark?
    non_spark_benchmark()

    spark, sqlContext = setup_spark_env()

    # How fast are the UDFs over the same data?
    single_benchmark(spark, sqlContext, mult=1, append_ids=False)

    # How does performance vary with the number of records?
    multiple_benchmarks(spark, sqlContext)

    # How fast are the UDFs over more data, picked to create partitions approaching 200MB
    single_benchmark(spark, sqlContext)


if __name__ == '__main__':

    main()
