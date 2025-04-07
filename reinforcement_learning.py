import numpy as np
import random
import matplotlib.pyplot as plt

equations = [
    (1, 1, 6),
    (2, -3, -4),
    (-1, 4, 5),
    (3, 2, 12),
    (-2, -1, -7)
]

actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

q_table = {}

learning_rate = 0.1
discount_factor = 0.95
exploration_rate = 1.0
exploration_decay = 0.995
num_episodes = 10000

def initialize_q_table():
    for x in range(-10, 11):
        for y in range(-10, 11):
            for eq_idx in range(len(equations)):
                q_table[(x, y, eq_idx)] = np.zeros(len(actions))

def get_reward(x, y, equation):
    a, b, target = equation
    prediction = a * x + b * y
    error = abs(prediction - target)
    return -error

def choose_action(x, y, eq_idx):
    if random.random() < exploration_rate:
        return random.randint(0, len(actions) - 1)
    return np.argmax(q_table[(x, y, eq_idx)])

def update_q(x, y, eq_idx, action_idx, reward, next_x, next_y):
    current_q = q_table[(x, y, eq_idx)][action_idx]
    max_future_q = np.max(q_table[(next_x, next_y, eq_idx)])
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
    q_table[(x, y, eq_idx)][action_idx] = new_q

def train():
    global exploration_rate
    initialize_q_table()
    
    for episode in range(num_episodes):
        x, y = random.randint(-10, 10), random.randint(-10, 10)
        eq_idx = random.randint(0, len(equations) - 1)
        total_reward = 0

        for _ in range(100):
            action_idx = choose_action(x, y, eq_idx)
            dx, dy = actions[action_idx]
            next_x = np.clip(x + dx, -10, 10)
            next_y = np.clip(y + dy, -10, 10)

            reward = get_reward(next_x, next_y, equations[eq_idx])
            update_q(x, y, eq_idx, action_idx, reward, next_x, next_y)

            x, y = next_x, next_y
            total_reward += reward

        exploration_rate *= exploration_decay

        if episode % 500 == 0:
            print(f"Episode {episode}/{num_episodes} - Reward: {total_reward:.2f}")

    print("Training complete.")

def test():
    print("\nTesting best solution for each equation:")
    for eq_idx, (a, b, target) in enumerate(equations):
        best_state = None
        min_error = float("inf")

        for x in range(-10, 11):
            for y in range(-10, 11):
                if (x, y, eq_idx) in q_table:
                    prediction = a * x + b * y
                    error = abs(prediction - target)
                    if error < min_error:
                        min_error = error
                        best_state = (x, y)

        x, y = best_state
        prediction = a * x + b * y
        print(f"Equation {a}x + {b}y = {prediction:.2f} (target: {target})")
        print(f"Best (x, y): ({x}, {y}) => Error: {abs(prediction - target):.2f}\n")

def interactive_test():
    print("🧪 Interactive equation testing (enter 'q' to quit)\n")
    while True:
        try:
            a = input("Enter coefficient a: ")
            if a.lower() == 'q': break
            b = input("Enter coefficient b: ")
            if b.lower() == 'q': break
            c = input("Enter target value: ")
            if c.lower() == 'q': break

            a, b, c = float(a), float(b), float(c)
            best_state = None
            min_error = float("inf")

            for x in range(-10, 11):
                for y in range(-10, 11):
                    prediction = a * x + b * y
                    error = abs(prediction - c)
                    if error < min_error:
                        min_error = error
                        best_state = (x, y)

            x, y = best_state
            print(f"\nBest (x, y): ({x}, {y}) -> {a}x + {b}y = {a*x + b*y:.2f} (target: {c})\n")
        except Exception as e:
            print(f"Error: {e}\nTry again.\n")

def show_graph(eq_idx):
    a, b, target = equations[eq_idx]
    error_grid = np.zeros((21, 21))  # from -10 to 10

    for i, x in enumerate(range(-10, 11)):
        for j, y in enumerate(range(-10, 11)):
            prediction = a * x + b * y
            error = abs(prediction - target)
            error_grid[j, i] = error  # (y, x) for image coordinates

    plt.figure(figsize=(6, 6))
    plt.imshow(error_grid, cmap='coolwarm', origin='lower', extent=[-10, 10, -10, 10])
    plt.colorbar(label='Error')
    plt.title(f"Error heatmap for equation {a}x + {b}y = {target}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

train()
test()
interactive_test()
show_graph(0)