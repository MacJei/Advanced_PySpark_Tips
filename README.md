## Advanced tips of pySpark

### Complicated condition in _when_ or _filter_
```python
from operator import and_, or_
from functools import reduce
import pyspark.sql.functions as F

def all_or(a_list):
	return reduce(or_, a_list)
	
reg_exp = {
  'languinis':['la%', '%lng%'],
  'operation new earth': ['one%'],
}

_cols = []
for game, reg_list in reg_exp.items():
  condition = all_or([F.lower(F.col('name')).like(reg_) for reg_ in reg_list])
  _cols.append(F.when(condition, F.lit(game)))
_cols[-1] = _cols[-1].otherwise(F.lit('others'))

data = data.withColumn('game', F.coalesce(*_cols))
```

### _udf_ with dynamic arguments
```python
import pyspark.sql.functions as F

def define_lt_udf(argument_list, game):
  if game in argument_list:
    return 'in_list'
  else:
    return 'out_list'
  
def define_lt(argument_list):
  return F.udf(lambda game: define_lt_udf(argument_list, game))

argument_list = ['languinis', 'operation new earth']
data = data.withColumn('if_in_list', define_lt(argument_list)(F.col('game')))
```

### _udf_ on _row_ level
```python
import pyspark.sql.functions as F
import pyspark.sql.types as T

my_udf = F.udf( lambda r: func(r), T.DoubleType())
data = data.withColumn(my_udf(F.struct([data[col] for col in data.columns])
```

### _pivot_ and _unpivot_
```python
import pyspark.sql.functions as F

for count_n in range(1,month_number):
  sub_str =  "key_col_val_{x}, key_col_name_{x}".format(x = count_n)
  sub_str_list.append(sub_str)
exprs = "stack({}, ".format(len(sub_str_list)+1) + ", ".join(sub_str_list)+") as (key_col, val_col)"
unpivot = data.select(*pivot_cols, F.expr(exprs))

pivot_table = unpivot.groupBy(group_cols).pivot(to_cols_col).sum(val_col)
```

### List of _agg_ operations
```python
# option 1
import pyspark.sql.functions as F

agg_ops = [F.sum(F.col('a')).alias('a'),
		F.countDistinct(F.when(F.col('a')>0, F.col('b'))).alias('distinct')]
df = df.groupBy(c).agg(agg_ops)

# option 2
def  _rename_after_agg(df, agg_ops):
	"""
	To rename the columns after aggregation.
	args:
		df (pyspark.DF): dataframe need to be renamed
		agg_ops (dict): {col: agg}
	return:
		df (pyspark.DF): renamed dataframe
	"""
	from functools import reduce
	renames = [("{}({})".format(v, k), k) for k, v in agg_ops.items()]
	df = reduce(lambda df, name: df.withColumnRenamed(name[0], name[1]), renames, df)
	return df
```

### Explode array
In case that you need to add missing rows related to a column.
```
A | B        A | B
--|----      --|----
1 | 'a'      1 | 'a'
3 | 'b'      2 | 'a'
         =>  3 | 'a'
             1 | 'a'
             2 | 'b'
             3 | 'b'                  
```
Or just need a continuous sequence.
```python
import pyspark.sql.functions as F
import pyspark.sql.types as T

n_to_array = F.udf(lambda x : list(range(x)), T.ArrayType(T.IntegerType()))

data = data.withColumn('int_list', n_to_array(F.col('int_col')))
data = data.withColumn('exploded_int_col', F.explode(F.col('int_list')))
```

### Forward / Backward fill
[reference](https://johnpaton.net/posts/forward-fill-spark/)
```python
from pyspark.sql import Window
from pyspark.sql.functions import F

# define the window
forward_fill_window = Window.partitionBy('group_col')\
               .orderBy('time')\
               .rowsBetween(-sys.maxsize, 0)
# forward-filled column
forward_filled_col = F.last(F.col('to_fill_col'), ignorenulls=True).over(forward_fill_window)
forward_filled_df = data.withColumn('forward_filled_col', forward_filled_col)
# define the window
backward_fill_window = Window.partitionBy('group_col')\
               .orderBy('time')\
               .rowsBetween(0, sys.maxsize)
# backward-filled column
backward_filled_col = F.first(F.col('to_fill_col'), ignorenulls=True).over(backward_fill_window)
backward_filled_df = data.withColumn('backward_filled_col', backward_filled_col)
```

### Some useful functions
```python
def all_add(a_list):
	"""
	To apply AND on all elements of list.
	args:
		a_list [boolean]: list of boolean elements
	"""
	from functools import reduce
	from operator import and_
	return reduce(and_, a_list)
	
def all_or(a_list):
	"""
	To apply OR on all elements of list.
	args:
		a_list [boolean]: list of boolean elements
	"""
	from functools import reduce
	from operator import or_
	return reduce(or_, a_list)
	
def union_all(*dfs):
	"""
	To union multiple tables.
	args:
		*dfs (*[pySpark.Dataframe]): list of tables
	"""
	from functools import reduce
	from pyspark.sql.dataframe import DataFrame
	return reduce(DataFrame.unionByName, dfs).cache()
	
def MSG(msg):
	"""
	To print log format information.
	args:
		msg (str)
	"""
	from datetime import datetime
	print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ':  '+ msg)

def update_by_merge(source, update, match_on, update_when_matched = True, update_sets = None, if_run = True, insert_not_match = True, show_stats = True):
	"""
	To use MERGE INTO functional of DataBricks.
	args:
		source (str):               table name of history data
		update (pyspark.DF):        new data
		match_on (list(str)):       list of columns used to match
		update_sets (list(str)):    list of columns need to update when match, if None, update all columns
		if_run (boolean):           if run the merge clause or just get the str
		insert_not_match (boolean): if insert new row when not match
		show_stats (boolean):        if show basic stats of the operation
	return:
		clause (str):             merge into clause
	"""
	assert isinstance(match_on,list), 'match_on needs to be a list'
	assert isinstance(update_sets,list) or update_sets is None, 'update_sets needs to be a list or None'

	original_source_n = spark.table(source).count() if (show_stats and if_run) else None

	update_table = source.split('.')[-1]+"_updates"
	update.createOrReplaceTempView(update_table)
	match_clause = ' and '.join([''.join([source,'.',col,' = ',update_table,'.',col]) for col in match_on])
	update_clause = ', '.join([''.join([col,' = ',update_table,'.',col]) for col in update_sets]) if update_sets else  '*'
	part_clause = ["MERGE INTO", source, "USING", update_table, 
  "\n ON", match_clause]
	if update_when_matched == True:
		part_clause.append("\nWHEN MATCHED THEN \n UPDATE SET")
		part_clause.append(update_clause)
	if insert_not_match:
		part_clause.append("\nWHEN NOT MATCHED THEN INSERT *")
	clause = ' '.join(part_clause)

	if if_run:
		sqlContext.sql(clause)

	if (show_stats and if_run):
		total_update_n = update.count()
		merged_source_n = spark.table(source).count()
		new_row_n = merged_source_n - original_source_n
		updated_row_n = total_update_n - new_row_n if insert_not_match else 'due to the un-insert not-match row, not sure how many'
		print("[log]: {new_row_n} rows are inserted, {updated_row_n} rows are updated (even by same value).".format(new_row_n=new_row_n, updated_row_n=updated_row_n))

	return clause
```
----------------------

## Some small thing which may kill you in pySpark

1.  The return of spark `udf` is very sensitive to __Data Type__, only use built-in type. E.g. `list()`, `int()`, `float()`
2.  When pyspark `sc.broadcast` data (variable) in defined function, both the __function__ and __args__ must be in __global namespace__.
Check error SPARK-5063: [explanation](https://stackoverflow.com/questions/31396323/spark-error-it-appears-that-you-are-attempting-to-reference-sparkcontext-from-a)
3.  When use `.asDict()` of Row, this return is __not sorted__.
4.  When use `.union()`, the column __order may not remain__.Check more [details](https://datascience.stackexchange.com/questions/11356/merging-multiple-data-frames-row-wise-in-pyspark). Use `.unionByName()`
5.  When use `sc.broadcast` multiple times on variable with same name, it will __not be updated__. [Details](http://apache-spark-user-list.1001560.n3.nabble.com/Broadcast-variables-can-be-rebroadcast-td22908.html). Some suggestions: 
	- [work advice](https://stackoverflow.com/questions/33372264/how-can-i-update-a-broadcast-variable-in-spark-streaming)
	```python
	for _ in range(n):
	  global broadcasted_var
	  broadcasted_var = sc.broadcast(local_var)
	  pyspark_df.withColumn('column', udf(broadcasted_var))
	  global broadcasted_var
      broadcasted_var.unpersist(True) # to remove cache in executors (workers)
      del broadcasted_var # to reomve cache on master (driver)
	```
	- Tried dynamic variable name using `global()`, but it doesn’t work, Because spark doesn’t take `global()[x]` as input, only direct variable name.
6.  `not` operator in .`fliter()` is not `!`, but `~`
7.  `filter((condition1) & (condition2))`, `()` of each condition is required
8. `+` between columns can not handle `None` value, other basic operators are the same. `operator.add` too.
9. `F.regexp_extract` will return empty string `''` when no match instead of `null`
10. When use `MERGE INTO` function of __DataBricks__, make sure all match_on columns have no null value, otherwise, it will insert new rows instead of update existing ones, because `null` and `null` doesn't match. The same when do `join` operation.
11. Do `select` before other operations will speed up a lot, especially when the source data are huge.
12. Clarify data type before comparison operation. In python `'0.9' > 0 == False` (in this case `'0.9'` is casted to __IntegerType__). Two solution: 1. `'0.9' > 0.0`; 2. cast StringType column into numeric.
