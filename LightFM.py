from lightfm import LightFM
from helper_funcs import (load_data, get_interaction_matrix_alt, 
    evaluate_lightfm, get_item_features, get_item_to_idx, save_results,
    save_preds)

train_path = "data/MINDlarge_train/"
dev_path = "data/MINDlarge_dev/"

behaviors_train = load_data(train_path,"behaviors")
news_train = load_data(train_path,"news")
behaviors_dev = load_data(dev_path,"behaviors")
news_test = load_data(dev_path,"news")

M,user_to_idx, item_to_idx_train = get_interaction_matrix_alt(behaviors_train,news_train)
item_to_idx_test = get_item_to_idx(news_test)

article_features_train = get_item_features(news_train)
article_features_test = get_item_features(news_test)

model = LightFM(no_components=50, loss='bpr',item_alpha= 1e-6)
model.fit(M, epochs=5, item_features=article_features_train, num_threads=16)
print("Calculating and saving Training Preds...")
# user_item_to_score_train = save_preds(model,user_to_idx,item_to_idx_train,article_features_train)

print("Calculating and saving Test Preds...")
# user_item_to_score_test = save_preds(model,user_to_idx,item_to_idx_test,article_features_test)

print("Evaluating on training data...")
training_res = evaluate_lightfm(model, behaviors_train, article_features_train, user_to_idx, item_to_idx_train)

print("Training Metrics:")
for metric, value in training_res.items():
    print(f"{metric}: {value}")

print("Evaluating on dev data...")
dev_res = evaluate_lightfm(model, behaviors_dev, article_features_test, user_to_idx, item_to_idx_test)

print("Development Metrics:")
for metric, value in dev_res.items():
    print(f"{metric}: {value}")

output_file = "res_LightFM.txt"
save_results(output_file,training_res,dev_res)

