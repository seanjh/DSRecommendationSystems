
# coding: utf-8

# In[1]:

import findspark
findspark.init()

from pyspark.mllib.recommendation import ALS, Rating
from pyspark import SparkContext, SQLContext
sc = SparkContext("local", "test")
sqlContext = SQLContext(sc)


# In[2]:

TRAIN_FILE = "./data//ratings-train.dat"
VALIDATION_FILE = "./data//ratings-validation.dat"
TEST_FILE = "./data/ratings-test.dat"


# In[3]:

def prepare_data(data):
    return (
        data
        .map(lambda l: l.split(','))
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    )  


# In[4]:

# Load and parse the data
ratings_train_text = sc.textFile(TRAIN_FILE)
ratings_train = prepare_data(ratings_train_text)


# In[5]:

ratings_validation_text = sc.textFile(VALIDATION_FILE)
ratings_validation = prepare_data(ratings_validation_text)


# In[6]:

ratings_test_text = sc.textFile(TEST_FILE)
ratings_test = prepare_data(ratings_validation_text)


# #### Calculate the general mean u for all ratings

# In[7]:

global_mean = ratings_train.map(lambda r: (r[2])).mean()


# In[8]:

global_mean


# ##### calculate item-specific bias, according to the paper we referenced, for each item i, its bias is equal to the summation of difference between all ratings of to the same item and global mean and then the result is divided by the sum of a regulation parameter and the quantity of the ratings.
# 

# In[11]:

df = sqlContext.createDataFrame(ratings_train, ['userId', 'movieId', 'ratings'])


# In[12]:

df_orderByMovie = df.orderBy(df.movieId)


# In[13]:

df_orderByMovie.take(5)


# In[29]:

movie_count = df_orderByMovie.groupBy(df_orderByMovie.movieId).count()


# In[60]:

movie_count.count()


# In[32]:

sum_byMovie = df_orderByMovie.groupBy(['movieId']).sum()


# In[45]:

drop_column1 = sum_byMovie.drop(sum_byMovie[1])


# In[46]:

final_drop = drop_column1.drop(drop_column1[1])


# In[47]:

final_drop.first()


# In[66]:

movie_sorted = movie_count.join(final_drop, "movieId")


# In[67]:

movie_sorted.take(5)


# In[68]:

new_movie_sorted = movie_sorted.orderBy(movie_sorted.movieId)


# In[69]:

new_movie_sorted.take(5)


# In[99]:

item_bias = new_movie_sorted.map(lambda r: [r[0], (r[2] - r[1]*global_mean)/(25+r[1])])


# In[100]:

new_item_bias = sqlContext.createDataFrame(item_bias, ['movieId', 'item_bias'])


# In[101]:

new_item_bias.take(10)


# In[87]:

item_bias.count()


# In[75]:

#calculate for user-specific bias
df_orderByUser = df.orderBy(df.userId)


# In[102]:

contain_itemBias = df_orderByUser.join(new_item_bias, "movieId")


# In[103]:

contain_itemBias.take(10)


# In[106]:

sorted_byUser = contain_itemBias.orderBy(['userId'])


# In[107]:

sorted_byUser.take(10)


# In[108]:

subtraction = sorted_byUser.map(lambda r: [r[1], r[2] - global_mean - r[3]])


# In[109]:

subtraction.take(10)


# In[110]:

user_bias_part1 = sqlContext.createDataFrame(subtraction, ['userId', 'subtraction'])


# In[111]:

sum_byUser = user_bias_part1.groupBy(['userId']).sum()


# In[112]:

sum_byUser.take(10)


# In[117]:

sum_UserCollect = user_bias_part1.groupBy(['userId']).count()


# In[119]:

ordered_sum_UserCollect = sum_UserCollect.orderBy(sum_UserCollect.userId)


# In[ ]:




# In[113]:

drop_column2 = sum_byUser.drop(sum_byUser[1])


# In[115]:

final_drop2 = drop_column2.orderBy(drop_column2.userId)


# In[116]:

final_drop2.take(10)


# In[120]:

user_bias_table = final_drop2.join(ordered_sum_UserCollect, 'userId')


# In[121]:

user_bias_table.take(5)


# In[122]:

ordered_userBiaTable = user_bias_table.orderBy(user_bias_table.userId)


# In[123]:

user_bias = ordered_userBiaTable.map(lambda r: [r[0], r[1]/(10+r[2])])


# In[124]:

user_bias.take(5)


# In[131]:

user_specific_bias = sqlContext.createDataFrame(user_bias, ['userId', 'user_bias'])


# In[132]:

user_specific_bias.take(10)


# In[127]:

df_orderByUser.take(10)


# In[133]:

merge1 = df_orderByUser.join(user_specific_bias, 'userId')


# In[134]:

merge1.take(5)


# In[135]:

merge2 = merge1.join(new_item_bias, 'movieId')


# In[136]:

merge2.take(5)


# In[137]:

new_ratings_train = merge2.map(lambda r: [r[0], r[1], r[2] - r[3] - r[4]])


# In[138]:

temp = sqlContext.createDataFrame(new_ratings_train, ['movieId', 'userId', 'new_ratings'])


# In[139]:

final_new_ratings_train = temp.orderBy(temp.userId)


# In[140]:

final_new_ratings_train.take(10)


# In[ ]:



