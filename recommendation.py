import math
import numpy as np
# =============================================================================
# common functionalities
# =============================================================================
def avg_current(test_list):
    sum_r = 0
    for u in test_list:
        sum_r += u
    return sum_r / 5


def clean_up(rating):
    if rating < 1:
        return 1
    if rating > 5:
        return 5
    return rating

# =============================================================================
# item based similarity functions
# =============================================================================
def find_cosine_sim_movies(movie1_list,movie2_list):
    numerator=0
    denominator,d1,d2=0,0,0
    for i in range(200):
        numerator+=movie1_list[i]*movie2_list[i]
        d1+=(movie1_list[i])**2
        d2+=(movie2_list[i])**2
    denominator=(d1**0.5)*(d2**0.5)
    return numerator/denominator

def sort_index(current_movie,similarity_list):
#    k=20
    similar_movies,movies=[],[]
    for index,val in enumerate(similarity_list):
        if index!=current_movie and val!=0:
            similar_movies.append((val,index))
    similar_movies.sort(reverse=True)
    
    for m in similar_movies:
        movies.append(m[1])
    return movies

def get_similar_movies(train_mat):
    cosine_movies=np.zeros(shape=(1000,1000))
    for i in range(1000):
        for j in range(1000):
            val=find_cosine_sim_movies(train_mat[:,i],train_mat[:,j])
            if not np.isnan(val):
                cosine_movies[i][j]=val
                
    similar_movies={k:[] for k in range(1000)}
    for index,row in enumerate(cosine_movies):
        similar_movies[index]=sort_index(index,row)
    return similar_movies,cosine_movies


def get_k_nearest(similar_movies,movie_index,cosine_movies,test_list):
    k=50
    nearest,newList=[],[]
    for m in similar_movies[movie_index]:
        newList.append((cosine_movies[movie_index][m],m))
        
    newList.sort(reverse=True)
    for m in newList:
        if len(nearest)<=k:
            nearest.append(m[1])
    return nearest
            
            
def item_based(movie_index,test_list,similar_movies,cosine_movies,train_mat):
    sim_m=get_k_nearest(similar_movies,movie_index,cosine_movies,test_list)
    rating_pred=0
    denomimnator,numerator=0,0
    for m in sim_m:
        if test_list[m]!=0:
            numerator+=cosine_movies[movie_index][m]*test_list[m]
            denomimnator+=cosine_movies[movie_index][m]
    if denomimnator!=0:
        rating_pred=math.ceil(numerator/denomimnator)
    else:
        rating_pred=math.ceil(avg_current(test_list))
    return clean_up(rating_pred)

# =============================================================================
# cosine similarity functions
# =============================================================================
def find_cosine_sim(curr_u, movie_index, test_list, train_mat):
    cosine, users, similarity = [], [], []
    k = 50
    for index, row in enumerate(train_mat):
        if row[movie_index] != 0:
            numerator, d1, d2 = 0, 0, 0
            for j, col in enumerate(row):
                if col != 0 and test_list[j] != 0:
                    numerator += col * test_list[j]
                    d1 += col ** 2
                    d2 += test_list[j] ** 2
            denominator = (d1 ** 0.5) * (d2 ** 0.5)
            if denominator != 0:
                cosine.append(((numerator / denominator), index))

    cosine.sort(reverse=True)
    if len(cosine) >= k:
        for i in range(k):
            users.append(cosine[i][1])
            similarity.append(cosine[i][0])
    else:
        for u in cosine:
            users.append(u[1])
            similarity.append(u[0])
    return similarity, users


def cosine_sim(curr_u, movie_index, test_list, train_mat):
    similarity, users = find_cosine_sim(curr_u, movie_index, test_list, train_mat)
    rating_pred = 0
    if len(users) == 0:
        rating_pred = math.ceil(avg_current(test_list))
    else:
        weight = 0
        for index, u in enumerate(users):
            weight += similarity[index] * train_mat[u][movie_index]
        rating_pred = math.ceil(weight / sum(similarity))
    return clean_up(rating_pred)

# =============================================================================
# pearson correkation functions
# =============================================================================
def find_avg_test(movie_index, test_list):
    avg_test, count = 0, 0
    for index, rating in enumerate(test_list):
        if index != movie_index:
            avg_test += rating
            count += 1
    avg_test /= count
    return avg_test


def avg_train_user(movie_index, train_mat):
    avg_user = {}
    for index, row in enumerate(train_mat):
        if row[movie_index] != 0:
            count, avg_u = 0, 0
            for movie, rating in enumerate(row):
                if movie != movie_index:
                    avg_u += rating
                    count += 1
            avg_user[index] = avg_u / count
    return avg_user


def find_weights(avg_user, train_mat, test_list, avg_test):
    weights = {}
    users = list(avg_user.keys())
    for user_index, user in enumerate(train_mat):
        if user_index in users:
            numerator, denominator = 0, 0
            d1, d2, n1, n2 = 0, 0, 0, 0
            for movie, rate in enumerate(user):
                n1 = rate - avg_user[user_index]
                n2 = test_list[movie] - avg_test
                numerator += n1 * n2
                d1 += n1 ** 2
                d2 += n2 ** 2
            denominator = (d1 ** 0.5) * (d2 ** 0.5)
            if denominator != 0:
                weights[user_index] = numerator / denominator
    return weights


def pearson_corelation(curr_u, movie_index, test_list, train_mat):
    # find avg_test for test user excluding movie_index
    avg_test = find_avg_test(movie_index, test_list)

    # find avgerage of each train user excluding  and remember relevant users
    avg_user = avg_train_user(movie_index, train_mat)
    sim_users = list(avg_user.keys())

    # find weights for each relevant users(= users who have rated movie_index)
    weights = find_weights(avg_user, train_mat, test_list, avg_test)
    
    rating_pred = 0
    if len(weights) == 0:
        rating_pred = math.ceil(avg_current(test_list))
    else:
        # find test users rating for movie_index
        numerator = 0
        for user_index, user in enumerate(train_mat):
            if user_index in sim_users:
                numerator += (user[movie_index] - avg_user[user_index]) * weights[user_index]
        denominator = 0
        for u in sim_users:
            denominator += abs(weights[u])    
            rating_pred = math.ceil(avg_test + (numerator / denominator))
   
    return clean_up(rating_pred)