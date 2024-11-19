# code
Tic tac toe 
import random

board = [" "] * 9

def print_board():
    for i in range(0, 9, 3): print(" | ".join(board[i:i+3])); print("-" * 5 if i < 6 else "")

def check_winner():
    for a, b, c in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
        if board[a] == board[b] == board[c] != " ": return board[a]

def human_move(): 
    while (move := int(input("Your move (1-9): ")) - 1) not in range(9) or board[move] != " ":
        print("Invalid move!")
    board[move] = "X"

def computer_move():
    board[random.choice([i for i, cell in enumerate(board) if cell == " "])] = "O"

print("Welcome to Tic-Tac-Toe!"); print_board()
while True:
    human_move(); print_board()
    if (winner := check_winner()) or " " not in board: break
    computer_move(); print("Computer's turn:"); print_board()
    if (winner := check_winner()) or " " not in board: break

print(f"{winner} wins!" if winner else "It's a tie!")


without random 
board = [" "] * 9

def print_board():
    for i in range(0, 9, 3): print(" | ".join(board[i:i+3])); print("-" * 5 if i < 6 else "")

def check_winner():
    for a, b, c in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
        if board[a] == board[b] == board[c] != " ": return board[a]

def human_move(): 
    while (move := int(input("Your move (1-9): ")) - 1) not in range(9) or board[move] != " ":
        print("Invalid move!")
    board[move] = "X"

def computer_move():
    for i in range(9):
        if board[i] == " ": board[i] = "O"; break

print("Welcome to Tic-Tac-Toe!"); print_board()
while True:
    human_move(); print_board()
    if (winner := check_winner()) or " " not in board: break
    computer_move(); print("Computer's turn:"); print_board()
    if (winner := check_winner()) or " " not in board: break

print(f"{winner} wins!" if winner else "It's a tie!")

n - queen problem 
def is_safe(board, row, col, n):
    for i in range(col):
        if board[row][i] == 1:
            return False
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    for i, j in zip(range(row, n, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    return True

def solve_nqueens(board, col, n):
    if col == n:
        return True
    for i in range(n):
        if is_safe(board, i, col, n):
            board[i][col] = 1
            if solve_nqueens(board, col + 1, n):
                return True
            board[i][col] = 0
    return False

def print_board(board, n):
    for i in range(n):
        print(" ".join("Q" if board[i][j] == 1 else "." for j in range(n)))

n = 9
board = [[0] * n for i in range(n)]
if solve_nqueens(board, 0, n):
    print_board(board, n)
else:
    print("Solution does not exist.")

Water-Jug
from collections import deque

def water_jug_bfs(jugs, target):
    visited = set()
    queue = deque([(tuple(0 for _ in jugs), [])])  # Start with all jugs empty
    
    while queue:
        state, path = queue.popleft()
        
        if state in visited:
            continue
        visited.add(state)
        
        # Add this step to the path
        path = path + [state]
        
        # Check if the target amount is in any jug
        if target in state:
            for step in path:
                print("State:", step)
            return state
        
        # Generate all possible moves
        for i in range(len(jugs)):
            # Fill jug i
            new_state = list(state)
            new_state[i] = jugs[i]
            queue.append((tuple(new_state), path))
            
            # Empty jug i
            new_state = list(state)
            new_state[i] = 0
            queue.append((tuple(new_state), path))
            
            # Pour from jug i to all other jugs
            for j in range(len(jugs)):
                if i != j:
                    new_state = list(state)
                    pour = min(state[i], jugs[j] - state[j])
                    new_state[i] -= pour
                    new_state[j] += pour
                    queue.append((tuple(new_state), path))
    
    print("No solution found.")
    return "No solution."

# Example usage
jugs = [int(x) for x in input("Enter jug capacities (space-separated): ").split()]
target = int(input("Enter the target amount: "))
result = water_jug_bfs(jugs, target)

if result != "No solution.":
    print("Final state:", result)
else:
    print(result)


TSP
from itertools import permutations

# A function to calculate the minimum cost path for the TSP
def tsp(graph, start):
    # Get the number of vertices
    n = len(graph)
    # Store all vertices except the start
    vertices = list(range(n))
    vertices.remove(start)
    # Store the minimum cost path and the route
    min_cost = float('inf')
    best_path = []
    
    # Generate all permutations of cities
    for perm in permutations(vertices):
        current_cost = 0
        current_route = [start] + list(perm) + [start]
        
        # Calculate the cost of the current route
        for i in range(len(current_route) - 1):
            current_cost += graph[current_route[i]][current_route[i+1]]
        
        # Check if the current route is the shortest
        if current_cost < min_cost:
            min_cost = current_cost
            best_path = current_route
    
    return min_cost, best_path

# Example graph (Adjacency matrix representing distances between cities)
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Starting from the first city (index 0)
start = 0
min_cost, best_path = tsp(graph, start)

# Output the result in a more user-friendly format
print("Minimum cost:", min_cost)
print("Best path:", " -> ".join(str(city) for city in best_path))


Wumpus world 

# Initialize the grid dynamically
grid_size = int(input("Enter grid size (e.g., 4 for a 4x4 grid): "))

# Define world entities dynamically
world = {}
agent_position = (1, 1)
agent_orientation = "E"  # Agent starts facing East
world[agent_position] = "A"

wumpus_position = tuple(map(int, input("Enter Wumpus position (x, y): ").split()))
pit_position = tuple(map(int, input("Enter Pit position (x, y): ").split()))
gold_position = tuple(map(int, input("Enter Gold position (x, y): ").split()))

world[wumpus_position] = "W"
world[pit_position] = "P"
world[gold_position] = "G"

# Function to display the world grid
def display_world():
    for y in range(grid_size, 0, -1):  # Print rows from top to bottom
        row = []
        for x in range(1, grid_size + 1):  # Print columns from left to right
            if agent_position == (x, y):  # Agent's position
                if agent_orientation == 'N':
                    row.append("A↑")
                elif agent_orientation == 'E':
                    row.append("A→")
                elif agent_orientation == 'S':
                    row.append("A↓")
                elif agent_orientation == 'W':
                    row.append("A←")
            elif (x, y) in world:  # Check if Wumpus, Pit, or Gold exists here
                row.append(world[(x, y)])
            else:
                row.append(".")  # Empty cell
        print(" | ".join(row))
    print()

# Function to move the agent
def move_agent():
    global agent_position, agent_orientation
    move = input("Enter move (forward, left, right): ").strip().lower()
    x, y = agent_position

    if move == "forward":
        if agent_orientation == "N" and y < grid_size:  # Move North
            agent_position = (x, y + 1)
        elif agent_orientation == "E" and x < grid_size:  # Move East
            agent_position = (x + 1, y)
        elif agent_orientation == "S" and y > 1:  # Move South
            agent_position = (x, y - 1)
        elif agent_orientation == "W" and x > 1:  # Move West
            agent_position = (x - 1, y)
        else:
            print("Bump! Can't move forward.")
    elif move == "left":
        turn_agent("left")
    elif move == "right":
        turn_agent("right")
    else:
        print("Invalid move.")
    
# Function to turn the agent
def turn_agent(direction):
    global agent_orientation
    orientations = ["N", "E", "S", "W"]  # North, East, South, West
    index = orientations.index(agent_orientation)
    if direction == "left":
        agent_orientation = orientations[(index - 1) % 4]
    elif direction == "right":
        agent_orientation = orientations[(index + 1) % 4]

# Game loop
while True:
    # Display the world grid
    display_world()
    print(f"Agent Position: {agent_position}")
    print(f"Agent Orientation: {agent_orientation}")

    # Check game conditions
    x, y = agent_position
    if (x, y) == wumpus_position:
        print("You encountered the Wumpus! Game over!")
        break
    elif (x, y) == pit_position:
        print("You fell into a pit! Game over!")
        break
    elif (x, y) == gold_position:
        print("You found the gold! You win!")
        break

    # Ask for the next move
    move_agent()


BAYESIAN CLASSIFICATION

# Given probabilities
P_Y = {0: 0.5, 1: 0.5}
P_X_given_Y = {0: 0.3, 1: 0.7}
P_Z_given_Y = {0: 0.4, 1: 0.6}

# Conditional probabilities for D
P_A = 0.75
P_B_given_A = 0.2
P_C_given_A = 0.7
P_D_given = {
    (1, 1): 0.3,  # P(D | B=1, C=1)
    (1, 0): 0.25, # P(D | B=1, C=0)
    (0, 1): 0.1,  # P(D | B=0, C=1)
    (0, 0): 0.35  # P(D | B=0, C=0)
}

# Calculate marginal probabilities for X and Z
P_X = sum(P_X_given_Y[y] * P_Y[y] for y in P_Y)
P_Z = sum(P_Z_given_Y[y] * P_Y[y] for y in P_Y)

# Calculate joint probability P(X=1, Z=1)
P_X_and_Z = sum(
    P_X_given_Y[y] * P_Z_given_Y[y] * P_Y[y]
    for y in P_Y
)

# Check independence
independent = abs(P_X_and_Z - (P_X * P_Z)) < 1e-6

# Calculate P(D | A)
P_B_and_C = P_B_given_A * P_C_given_A
P_B_and_not_C = P_B_given_A * (1 - P_C_given_A)
P_not_B_and_C = (1 - P_B_given_A) * P_C_given_A
P_not_B_and_not_C = (1 - P_B_given_A) * (1 - P_C_given_A)

P_D_given_A = (
    P_D_given[(1, 1)] * P_B_and_C +
    P_D_given[(1, 0)] * P_B_and_not_C +
    P_D_given[(0, 1)] * P_not_B_and_C +
    P_D_given[(0, 0)] * P_not_B_and_not_C
)

# Output the results
print(f"P(X=1) = {P_X:.2f}")
print(f"P(Z=1) = {P_Z:.2f}")
print(f"P(X=1, Z=1) = {P_X_and_Z:.2f}")
print(f"P(X=1) * P(Z=1) = {P_X * P_Z:.2f}")
print(f"Are X and Z independent? {'Yes' if independent else 'No'}")
print(f"The probability of D being true given that A is true is: {P_D_given_A:.3f}")


DECISION TREE

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Define the dataset
data = {
    "Weather": ["sunny", "sunny", "windy", "rainy", "rainy", "rainy", "windy", "windy", "windy", "sunny", "sunny"],
    "Parents": ["visit", "no-visit", "visit", "visit", "no-visit", "visit", "no-visit", "no-visit", "visit", "no-visit", "no-visit"],
    "Cash": ["rich", "rich", "rich", "poor", "rich", "poor", "poor", "rich", "rich", "rich", "poor"],
    "Exam": ["yes", "no", "no", "yes", "no", "no", "yes", "yes", "no", "no", "yes"],
    "Decision": ["cinema", "tennis", "cinema", "cinema", "stay-in", "cinema", "cinema", "shopping", "cinema", "tennis", "tennis"]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Map categorical variables to numerical values
mappings = {}

for column in df.columns:
    if df[column].dtype == 'object':
        unique_values = df[column].unique()
        mappings[column] = {val: idx for idx, val in enumerate(unique_values)}
        df[column] = df[column].map(mappings[column])

# Separate features and target variable
X = df[["Weather", "Parents", "Cash", "Exam"]]
y = df["Decision"]

# Train a decision tree classifier
decision_tree = DecisionTreeClassifier(criterion="entropy", random_state=42)
decision_tree.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(
    decision_tree,
    feature_names=["Weather", "Parents", "Cash", "Exam"],
    class_names=list(mappings["Decision"].keys()),
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Full Decision Tree")
plt.show()

# Inverse mapping function to interpret outputs
inverse_mappings = {col: {v: k for k, v in mappings[col].items()} for col in mappings}

# Test the decision tree on a sample input
def predict_decision(input_data):
    # Map input data to numerical values
    encoded_input = [mappings[col][input_data[col]] for col in input_data]
    prediction = decision_tree.predict([encoded_input])[0]
    return inverse_mappings["Decision"][prediction]



# Test the function
sample_input = {"Weather": "rainy", "Parents": "visit", "Cash": "rich", "Exam": "yes"}
print("Predicted Decision:", predict_decision(sample_input)) 

import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Define the training data based on the table
# L: 0 = short, 1 = long
# M: 0 = less math, 1 = more math
# D: 0 = easy (-), 1 = difficult (+)
data = [
    [0, 0, 0, 4],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 1, 3],
    [1, 0, 0, 1],
    [1, 0, 1, 2],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
]

# Expand the data by repeating rows according to the count (#)
expanded_data = []
for row in data:
    for _ in range(row[3]):  # Repeat row[3] times
        expanded_data.append(row[:3])  # Only L, M, D columns

# Convert to NumPy array
expanded_data = np.array(expanded_data)

# Separate features (L, M) and target (D)
X = expanded_data[:, :2]  # Features: L and M
y = expanded_data[:, 2]   # Target: D

# Train a decision tree classifier
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(8, 6))
plot_tree(clf, feature_names=["L", "M"], class_names=["Easy (-)", "Difficult (+)"], filled=True, rounded=True)
plt.title("Decision Tree for Problem Difficulty")
plt.show()

# Test the decision tree on a new problem
def predict_difficulty(L, M):
    prediction = clf.predict([[L, M]])[0]
    return "Difficult (+)" if prediction == 1 else "Easy (-)"

# Example predictions
print("Prediction for L=0, M=0:", predict_difficulty(0, 0))
print("Prediction for L=1, M=1:", predict_difficulty(1, 1))
print("Prediction for L=0, M=1:", predict_difficulty(0, 1))









#Decision Trees 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report 
 
X, y = [], [] 
num_samples = int(input("Enter number of samples: ")) 
num_features = int(input("Enter number of features: ")) 
     
for i in range(num_samples): 
    features = list(map(float, input(f"Enter features for sample {i + 1} (space separated):").split())) 
    X.append(features) 
 
    label = int(input(f"Enter label for sample {i + 1} (0 or 1):")) 
    y.append(label) 
         
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32) 
 
model = DecisionTreeClassifier() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 
classification_rep = classification_report(y_test, y_pred, zero_division=1) 
 
print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%") 
print("Classification Report:") 
print(classification_rep)





SENTIMENT ANALYSIS

from textblob import TextBlob

n = int(input("Number of reviews: "))
review_text = []
sentiment = []
for _ in range(n):
    review_text.append(input("Enter review text: "))
    sentiment.append(int(input("Enter sentiment (0 for negative, 1 for positive): ")))

def classify_sentiment(review):
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 1
    else:
        return 0

predicted_sentiments = [classify_sentiment(review) for review in review_text]

for review, predicted, actual in zip(review_text, predicted_sentiments, sentiment):
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {'Positive' if predicted == 1 else 'Negative'}")
    print(f"Actual Sentiment: {'Positive' if actual == 1 else 'Negative'}")
    print()





















import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Original dataset with reviews
data = {
    'review': [
        "Amazing storyline and great acting!",
        "Terrible pacing, I lost interest halfway.",
        "Beautiful cinematography and inspiring plot.",
        "Predictable plot with weak character development.",
        "Fantastic performance by the lead actor!",
        "Not worth the hype, found it quite boring.",
        "A delightful experience with stunning visuals.",
        "Poor dialogue and lack of suspense throughout.",
        "Heartwarming and beautifully crafted story.",
        "Too slow and uneventful to keep my attention.",
        "Absolutely fantastic! Loved it from start to finish.",
        "Horrible movie. The acting was bad, and the plot made no sense.",
        "What a waste of time! This was utterly boring.",
        "Brilliantly executed and very entertaining.",
        "Uninspired story and dull performances.",
        "Captivating and very well-directed.",
        "Mediocre at best. It felt like a chore to watch.",
        "A masterclass in storytelling. Simply beautiful.",
        "Forgettable experience. Wouldn't recommend.",
        "An engaging and exciting plot with stellar acting."
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative
}

# Converting the dataset into a DataFrame
df = pd.DataFrame(data)

# Function to clean the text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()  # Remove non-alphabetic characters
    text = text.split()  # Tokenize
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]  # Remove stopwords
    return ' '.join(text)

# Applying the text cleaning function to the dataset
df['cleaned_text'] = df['review'].apply(preprocess_text)

# Feature extraction using TF-IDF Vectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)  # Use bigrams and include all terms
X = tfidf.fit_transform(df['cleaned_text'])  # Features (text)
y = df['sentiment']  # Labels (positive or negative)

# Splitting the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'max_iter': [100, 500, 1000]
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Making predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluating the model
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# Function to predict sentiment for new input text
def predict_sentiment(new_text):
    cleaned_new_text = preprocess_text(new_text)
    vectorized_text = tfidf.transform([cleaned_new_text])
    prediction = best_model.predict(vectorized_text)
    return "Positive" if prediction[0] == 1 else "Negative"

# Testing the function with new examples
print("Sentiment of 'I loved the experience':", predict_sentiment('I loved the experience'))
print("Sentiment of 'The movie was terrible':", predict_sentiment('The movie was terrible'))

pip install numpy pandas matplotlib scikit-learn textblob nltk


