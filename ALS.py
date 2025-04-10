import implicit
from helper_funcs import (load_data,get_interaction_matrix,evaluate,save_results)

train_path = "data/MINDlarge_train/"
dev_path = "data/MINDlarge_dev/"


behaviors_train = load_data(train_path,"behaviors")
news_train = load_data(train_path,"news")
behaviors_dev = load_data(dev_path,"behaviors")
print("Data loaded. Building interaction matrix...")
M, user_to_idx, item_to_idx = get_interaction_matrix(behaviors_train,news_train)

print(f"Constructed and loaded user-item interaction matrix of shape {M.shape}")

model = implicit.als.AlternatingLeastSquares(
    factors = 50, regularization=0.1,iterations=10,num_threads = 4
)

matrix_conf = M*40 #multiply by alpha=40
print("Fitting model...")
model.fit(matrix_conf)

print("Evaluating on training data...")
training_res = evaluate(model, behaviors_train, user_to_idx, item_to_idx)

print("Training Metrics:")
for metric, value in training_res.items():
    print(f"{metric}: {value}")

print("Evaluating on dev data...")
dev_res = evaluate(model, behaviors_dev, user_to_idx, item_to_idx)

print("Development Metrics:")
for metric, value in dev_res.items():
    print(f"{metric}: {value}")

output_file = "res_ALS.txt"
save_results(output_file,training_res,dev_res)