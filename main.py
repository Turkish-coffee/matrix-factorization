import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class MF():
    
    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))
        
        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)
    


import tensorflow as tf
import json
import pandas as pd

with open('data/ranking.json', encoding="utf-8-sig") as f:
    data = json.load(f)

columns = ["Id","ProductId","BusinessId","UserId","Rate",
        "Family","Parent","Taste","Type","Kitchen","DescriptionTags"]
df = pd.DataFrame(data, columns=columns)

df = df.drop(df[df['UserId'] == 2123].index)
df = df.drop(df[df['UserId'] == 2245].index)
df = df.drop(df[df['UserId'] == 2260].index)


#print(df.head(2))


users = list(df["UserId"])
products = list(df["ProductId"])
ratings = list(df["Rate"].astype(int))


unique_products_list = list(df["ProductId"].unique())
unique_users_list = list(df["UserId"].unique())
unique_product_id_list = list(df["ProductId"].unique())

print(unique_users_list)
input()
#print(unique_products_list)

#print (len(unique_products_list))
#print (unique_products_list)
#print (len(unique_users_list))

# print(users[:2],products[:2],descriptionTags[:2],ratings[:2] )


# --------------------- defining some variables -----------------------

num_users = len(unique_users_list)
num_products = len(unique_products_list)
num_feats = len(df["DescriptionTags"][0])
num_recommendations = 10


# ---------------------here is the user_movies matrix-----------------------
users_products = []

for user in unique_users_list:
    temp = []
    for product in zip(users, ratings, unique_products_list):
        if user == product[0]:
            temp.append(product[1])        
        else:
            temp.append(0)
    users_products.append(temp)

users_products = tf.constant(users_products, dtype=tf.float32)
print(users_products)
input("users_products")

R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

R = np.array(users_products)

mf = MF(R, K=2, alpha=0.1, beta=0.01, iterations=20)
training_process = mf.train()
print()
print("P x Q:")
print(mf.full_matrix())
print()
print("Global bias:")
print(mf.b)
print()
print("User bias:")
print(mf.b_u)
print()
print("Item bias:")
print(mf.b_i)

x = [x for x, y in training_process]
y = [y for x, y in training_process]
plt.figure(figsize=((16,4)))
plt.plot(x, y)
plt.xticks(x, x)
plt.xlabel("Iterations")
plt.ylabel("Mean Square Error")
plt.grid(axis="y")
plt.show()
