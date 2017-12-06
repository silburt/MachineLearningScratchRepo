#Use large or small datasets
ratings_raw_RDD = sc.textFile('ratings.csv')
# ratings_raw_RDD = sc.textFile('ratings-large.csv')

#Parse lines
ratings_RDD = ratings_raw_RDD.map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2])))

#Split into training, validation and test sets
training_RDD, validation_RDD, test_RDD = ratings_RDD.randomSplit([3, 1, 1], 0)

#Create prediction sets without ratings
predict_validation_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
predict_test_RDD = test_RDD.map(lambda x: (x[0], x[1]))

from pyspark.mllib.recommendation import ALS
import math

seed = 5
iterations = 10
regularization = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0

min_error = float('inf')
best_rank = -1
best_iteration = -1

for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations, lambda_=regularization)
    #Coercing ((u,p),r) tuple format to accomodate join
    predictions_RDD = model.predictAll(predict_validation_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    ratings_and_preds_RDD = validation_RDD.map(lambda r: ((r[0], r[1]), r[2])).join(predictions_RDD)
    error = math.sqrt(ratings_and_preds_RDD.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < min_error:
        min_error = error
        best_rank = rank

print 'The best model was trained with rank %s' % best_rank

# Redo the last phase with the best rank size and using test dataset this time
model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations, lambda_=regularization)
predictions_RDD = model.predictAll(predict_test_RDD).map(lambda r: ((r[0], r[1]), r[2]))
ratings_and_preds_RDD = test_RDD.map(lambda r: ((r[0], r[1]), r[2])).join(predictions_RDD)
error = math.sqrt(ratings_and_preds_RDD.map(lambda r: (r[1][0] - r[1][1])**2).mean())
print 'For testing data the RMSE is %s' % (error)

# Add a new user. Assuming ID 0 is unused, but can check with "ratings_RDD.filter(lambda x: x[0]=='0').count()"
new_user_ID = 0
new_user = [
     (0,100,4), # City Hall (1996)
     (0,237,1), # Forget Paris (1995)
     (0,44,4), # Mortal Kombat (1995)
     (0,25,5), # etc....
     (0,456,3),
     (0,849,3),
     (0,778,2),
     (0,909,3),
     (0,478,5),
     (0,248,4)
    ]
new_user_RDD = sc.parallelize(new_user)
updated_ratings_RDD = ratings_RDD.union(new_user_RDD)

#Update model. This takes time.
updated_model = ALS.train(updated_ratings_RDD, best_rank, seed=seed, iterations=iterations, lambda_=regularization)

#Use large or small datasets
movies_raw_RDD = sc.textFile('movies.csv')
# movies_raw_RDD = sc.textFile('movies-large.csv')

#Parse lines
movies_RDD = movies_raw_RDD.map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1]))

#Create prediction type RDD of all movies not yet rated by new user
new_user_rated_movie_ids = map(lambda x: x[1], new_user)
new_user_unrated_movies_RDD = movies_RDD.filter(lambda x: x[0] not in new_user_rated_movie_ids).map(lambda x: (new_user_ID, x[0]))

#Get recomendations
new_user_recommendations_RDD = updated_model.predictAll(new_user_unrated_movies_RDD)

# Transform into (Movie ID, Predicted Rating)
#First turn from spark model struct to (produce,rating)
product_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
#Now join with Movies to get real title
new_user_recommendations_titled_RDD = product_rating_RDD.join(movies_RDD)
#In final format (movie,rating)
new_user_recommendations_formatted_RDD = new_user_recommendations_titled_RDD.map(lambda x: (x[1][1],x[1][0]))

#Top recommedations
top_recomends = new_user_recommendations_formatted_RDD.takeOrdered(10, key=lambda x: -x[1])
for line in top_recomends: print line

one_movie_RDD = sc.parallelize([(0, 800)]) # Lone Star (1996)
rating_RDD = updated_model.predictAll(one_movie_RDD)
rating_RDD.take(1)
