import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import implicit
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

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

    
def get_item_to_idx(n):
    unique_articles = n["news_id"].unique()
    item_to_idx = {A:i for i,A in enumerate(unique_articles)}
    return item_to_idx


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

def extract_entity_labels(entity_str):
    """Extracts and formats entity labels from a JSON string."""
    if pd.isnull(entity_str) or entity_str.strip() == "":
        return ""
    try:
        # Some datasets use single quotes, so replace with double quotes for valid JSON.
        entities = json.loads(entity_str.replace("'", "\""))
        # Extract the "Label" and prepend a tag (e.g., "entity:") to differentiate it.
        labels = ["entity:" + ent["Label"].replace(" ", "_") for ent in entities]
        return " ".join(labels)
    except Exception:
        return ""

def get_item_features(df):
    # Step 2. Process categorical features.
    # Create new string features for category and subcategory 
    df['category_feature'] = "cat:" + df["category"].astype(str)
    df['subcategory_feature'] = "subcat:" + df["subcategory"].astype(str)

    # Create a list of dictionaries (each item’s categorical features) for one-hot encoding.
    categorical_features = df[['category_feature', 'subcategory_feature']].to_dict(orient='records')

    # Use a DictVectorizer to one-hot encode the categorical features.
    dict_vec = DictVectorizer(sparse=True)
    cat_feature_matrix = dict_vec.fit_transform(categorical_features)

    # Step 3. Process text features.
    # Use CountVectorizer for title and abstract. These will produce bag-of-words representations.
    # Fill in any missing values as empty strings.
    cv_title = CountVectorizer()
    title_matrix = cv_title.fit_transform(df["title"].fillna(""))

    cv_abstract = CountVectorizer()
    abstract_matrix = cv_abstract.fit_transform(df["abstract"].fillna(""))

    # Create new features for entities.
    df['title_entity_features'] = df["title_entities"].apply(extract_entity_labels)
    df['abstract_entity_features'] = df["abstract_entities"].apply(extract_entity_labels)

    # Combine entity strings from both title and abstract entities.
    entity_text = df['title_entity_features'] + " " + df['abstract_entity_features']
    cv_entities = CountVectorizer()
    entity_feature_matrix = cv_entities.fit_transform(entity_text)

    # Step 5. Combine all features into one CSR matrix.
    # hstack stacks matrices horizontally (columns are concatenated).
    # You can choose to omit certain matrices if you decide a feature set is not needed.
    item_features = hstack([cat_feature_matrix, title_matrix, abstract_matrix, entity_feature_matrix]).tocsr()

    # Now, item_features is a (n_items, n_item_features) CSR sparse matrix ready for LightFM.
    print("Shape of item features matrix:", item_features.shape)
    return item_features


def get_interaction_matrix_alt(b,n):
    """ Create a sparse matrix where the data in row i and column j is 
    defined by data[a] = row[a] col[a]

    THIS IMPLEMENTATION REPLACES NON-CLICKS (0 WITH NEGATIVE SIGNAL -1)

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
            n_id,label = impression.split("-")[0],int(impression.split("-")[1])
            # if not label:
            #     label == -1
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
           - "impressions": a string of space-separated impressions. Each impression is of the form "newsID-label",
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

        # Split the "impressions" field: assume each impression is "newsID-label"
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

def save_preds(mod,user_ids,item_ids,item_features):
    
    # Generate all pairs (user, item) vectorized.
    # For example, if there are U users and I items, these arrays will be of shape (U * I,).
    # Option 1: Using repeat and tile.
    user_ids = np.repeat(user_ids_arr, len(item_ids_arr))
    item_ids = np.tile(item_ids_arr, len(user_ids_arr))
    
    # Option 2 (alternative): Using meshgrid.
    # u, i = np.meshgrid(user_ids_arr, item_ids_arr, indexing='ij')
    # user_ids = u.ravel()
    # item_ids = i.ravel()
    
    print("Making predictions...")
    # Predict on all user-item pairs at once. This call uses 16 threads as specified.
    scores = mod.predict(user_ids, item_ids, item_features=item_features, num_threads=16)
    
    # Build a dictionary mapping (user_index, item_index) pairs to predicted scores.
    user_item_to_score = {(user_ids[i], item_ids[i]): score for i, score in enumerate(scores)}
    return user_item_to_score

def evaluate_lightfm(mod, b, article_features, user_to_idx, item_to_idx):
    batch_size = len(b)//100
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    filtered_labels, user_idxs,item_idxs,cnts = [],[],[],[]

    for i, row in tqdm(b.iterrows(), total=len(b)):
        #for every 100th piece of data calculate scores and metrics
        if (i%batch_size == 0 and i > 0) or (i == len(b)-1):
            scores = mod.predict(user_idxs, item_idxs, item_features=article_features, num_threads=16)
            offset = 0
            for cnt in cnts:
                try:
                    aucs.append(roc_auc_score(filtered_labels[offset:offset+cnt], scores[offset:offset+cnt]))
                except Exception as e:
                    pass
                mrrs.append(mrr_score(filtered_labels[offset:offset+cnt], scores[offset:offset+cnt]))
                ndcg5s.append(ndcg_score(filtered_labels[offset:offset+cnt], scores[offset:offset+cnt], k=5))
                ndcg10s.append(ndcg_score(filtered_labels[offset:offset+cnt], scores[offset:offset+cnt], k=10))
                offset += cnt
            filtered_labels, user_idxs,item_idxs,cnts = [],[],[],[]

        user = row["user_id"]
        if user not in user_to_idx:
            continue

        user_idx = user_to_idx[user]
        impressions = row["impressions"].split()
        news_ids = [imp.split("-")[0] for imp in impressions]
        labels = [int(imp.split("-")[1]) for imp in impressions]
        filtered_pairs = [(nid, label) for nid, label in zip(news_ids, labels) if nid in item_to_idx]
        
        # If there are fewer than 2 valid interactions or all labels are zero, skip this row
        if len(filtered_pairs) < 2 or sum(label for _, label in filtered_pairs) == 0:
            continue

        cnts.append(len(filtered_pairs))
        user_ids = [user_idx]*len(filtered_pairs)
        labels = [label for nid, label in filtered_pairs]
        item_indices = [item_to_idx[nid] for nid, _ in filtered_pairs]
        
        filtered_labels += labels
        user_idxs += user_ids
        item_idxs += item_indices

    return {
    "AUC": np.mean(aucs) if aucs else None,
    "MRR": np.mean(mrrs) if mrrs else None,
    "nDCG@5": np.mean(ndcg5s) if ndcg5s else None,
    "nDCG@10": np.mean(ndcg10s) if ndcg10s else None
    }
         
    
    
# def evaluate_lightfm(mod, b, article_features, user_to_idx, item_to_idx):
#     """ 
#     Idea is break dataframe into 100 segments, for each segemnt accumulate indices of users
#     and their corresponding interacted articles, since predict needs pairs of users and items
#     egs: the user[i],item[i] would be the id of the user interacting with article at i
#     Hence also need to keep track of number of items per impression to allocate score to impression
#     then will be able to calculate ranking metrics and average over impressions

#     """
#     batch_size = len(b)//100
#     aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
#     user_idxs,item_idxs,cnts = [],[],[]
        
#     # Iterate over each row in the user behavior DataFrame.
#     for i, row in tqdm(b.iterrows(), total=len(b)):
#         if (i%batch_size == 0 and i > 0) or (i == len(b)-1):
#             user_item_to_score = save_preds()
#         user = row["user_id"]
#         if user not in user_to_idx:
#             continue
#         user_idx = user_to_idx[user]
#         user_idxs.append(user_idx)

#         # Split the "impressions" field: assume each impression is "newsID-label"
#         impressions = row["impressions"].split()
#         cnts.append(len(impressions))
#         news_ids = [imp.split("-")[0] for imp in impressions]
#         labels = [int(imp.split("-")[1]) for imp in impressions]
        
#         # Keep only impressions where the news_id exists in item_to_idx
#         # Zip together news_ids and labels and filter.
#         filtered_pairs = [(nid, label) for nid, label in zip(news_ids, labels) if nid in item_to_idx]
        
#         # If there are fewer than 2 valid interactions or all labels are zero, skip this row
#         if len(filtered_pairs) < 2 or sum(label for _, label in filtered_pairs) == 0:
#             continue
        
#         # Separate filtered labels and determine corresponding item indices.
#         filtered_labels = [label for nid, label in filtered_pairs]
#         item_indices = [item_to_idx[nid] for nid, _ in filtered_pairs]
#         item_idxs += item_indices

#         scores = [user_item_to_score[(user_idx,i)] for i in item_indices]
        
#         try:
#             aucs.append(roc_auc_score(filtered_labels, scores))
#         except Exception as e:
#             pass
        
#         mrrs.append(mrr_score(filtered_labels, scores))
#         ndcg5s.append(ndcg_score(filtered_labels, scores, k=5))
#         ndcg10s.append(ndcg_score(filtered_labels, scores, k=10))
    
#     return {
#         "AUC": np.mean(aucs) if aucs else None,
#         "MRR": np.mean(mrrs) if mrrs else None,
#         "nDCG@5": np.mean(ndcg5s) if ndcg5s else None,
#         "nDCG@10": np.mean(ndcg10s) if ndcg10s else None
#     }

def save_results(output_file,train_res,dev_res):
    with open(output_file, "w") as f:
        f.write("Training Evaluation Metrics:\n")
        for metric, value in train_res.items():
            f.write(f"{metric}: {value}\n")
        f.write("\nDevelopment Evaluation Metrics:\n")
        for metric, value in dev_res.items():
            f.write(f"{metric}: {value}\n")

    print(f"Evaluation results written to {output_file}")
