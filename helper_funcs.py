import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import implicit

# ---------------------
# Load interactions from behaviors.tsv
# ---------------------

def load_data(path, dataset):
    if dataset not in ["behaviors","news"]:
        raise ValueError("dataset has wrong value!")
    print(f"Loading {dataset} from {path}...")
    if dataset == "behaviors":
        return pd.read_csv(path+"behaviors.tsv", sep='\t', header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"])
    else:
        return pd.read_csv(path+"news.tsv",sep="\t",header=None,names=["news_id", "category", "subcategory", "title", "abstract",
                          "url", "title_entities", "abstract_entities"])

    
def get_interaction_matrix(b,n):
    """ Create a sparse matrix where the data in row i and column j is 
    defined by data[a] = row[a] col[a]

    the utility matrix defines the interaction of a user u with a
    news article n
    """
    unique_users = b["user_id"].unique()
    unique_articles = n["news_id"].unique()
    user_to_idx = {u:i for i,u in enumerate(unique_users)}
    item_to_idx = {A:i for i,A in enumerate(unique_articles)}
    user_interactions = []
    for _,row in tqdm(b.iterrows(), total=len(b)):
        u_id = row["user_id"]
        for impression in row["impressions"].split():
            n_id,label = impression.split("-")[0],float(impression.split("-")[1])
            user_interactions.append((u_id,n_id,label,user_to_idx[u_id],item_to_idx[n_id]))
    user_interactions = pd.DataFrame(user_interactions,
        columns = ["user_id","news_id","label","user_idx","news_idx"])
    print("Creating interaction matrix...")
    matrix = csr_matrix((user_interactions["label"],
        (user_interactions["user_idx"],user_interactions["news_idx"])),
        shape = (len(user_to_idx), len(item_to_idx)))
    return matrix, user_to_idx, item_to_idx
    
    


# ---------------------
# Ranking Metrics
# ---------------------
def mrr_score(labels, scores):
    ranked = np.argsort(scores)[::-1]
    for rank, idx in enumerate(ranked):
        if labels[idx] == 1:
            return 1.0 / (rank + 1)
    return 0.0

def ndcg_score(labels, scores, k):
    ranked = np.argsort(scores)[::-1][:k]
    dcg = sum([labels[i] / np.log2(rank + 2) for rank, i in enumerate(ranked)])
    ideal_dcg = sum([1.0 / np.log2(i + 2) for i in range(min(sum(labels), k))])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

# ---------------------
# Evaluation Function
# ---------------------

def evaluate(mod, b, user_to_idx, item_to_idx):
    """
    Evaluate the model's latent factors on user-news interactions.
    
    Parameters:
        mod: Model object with attributes:
             - mod.user_factors: a 2D array-like of user latent factors.
             - mod.item_factors: a 2D array-like of item latent factors.
        b: A pandas DataFrame of user behaviors with at least two columns:
           - "user_id": the ID of the user.
           - "impression": a string of space-separated impressions. Each impression is of the form "newsID-label",
                           where label is an integer (e.g. 1 for positive interaction).
        user_to_idx: A dictionary mapping each user_id to a row index.
        item_to_idx: A dictionary mapping each news_id to a column index.
        
    Returns:
        A dictionary of average evaluation metrics: AUC, MRR, nDCG@5, and nDCG@10.
    """
    
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    
    # Iterate over each row in the user behavior DataFrame.
    for _, row in tqdm(b.iterrows(), total=len(b)):
        user = row["user_id"]
        if user not in user_to_idx:
            continue
        user_idx = user_to_idx[user]

        # Split the "impression" field: assume each impression is "newsID-label"
        impressions = row["impressions"].split()
        news_ids = [imp.split("-")[0] for imp in impressions]
        labels = [int(imp.split("-")[1]) for imp in impressions]
        
        # Keep only impressions where the news_id exists in item_to_idx
        # Zip together news_ids and labels and filter.
        filtered_pairs = [(nid, label) for nid, label in zip(news_ids, labels) if nid in item_to_idx]
        
        # If there are fewer than 2 valid interactions or all labels are zero, skip this row
        if len(filtered_pairs) < 2 or sum(label for _, label in filtered_pairs) == 0:
            continue
        
        # Separate filtered labels and determine corresponding item indices.
        filtered_labels = [label for nid, label in filtered_pairs]
        item_indices = [item_to_idx[nid] for nid, _ in filtered_pairs]
        
        # Compute predicted scores from the latent factors using a dot product:
        # mod.user_factors[user_idx] is a vector,
        # mod.item_factors[item_indices] creates a 2D array (one row per valid impression).
        scores = np.dot(mod.user_factors[user_idx], mod.item_factors[item_indices].T)
        
        # Compute metrics; handle potential exceptions for roc_auc_score
        try:
            aucs.append(roc_auc_score(filtered_labels, scores))
        except Exception as e:
            # Optionally log the error message e
            pass
        
        mrrs.append(mrr_score(filtered_labels, scores))
        ndcg5s.append(ndcg_score(filtered_labels, scores, k=5))
        ndcg10s.append(ndcg_score(filtered_labels, scores, k=10))
    
    # Return aggregated metric averages; if any metric list is empty, return None for that metric.
    return {
        "AUC": np.mean(aucs) if aucs else None,
        "MRR": np.mean(mrrs) if mrrs else None,
        "nDCG@5": np.mean(ndcg5s) if ndcg5s else None,
        "nDCG@10": np.mean(ndcg10s) if ndcg10s else None
    }

def save_results(output_file,train_res,dev_res):
    with open(output_file, "w") as f:
        f.write("Training Evaluation Metrics:\n")
        for metric, value in train_res.items():
            f.write(f"{metric}: {value}\n")
        f.write("\nDevelopment Evaluation Metrics:\n")
        for metric, value in dev_res.items():
            f.write(f"{metric}: {value}\n")

    print(f"Evaluation results written to {output_file}")
