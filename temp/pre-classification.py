import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load engineered dataset
INPUT_FILE = "asrs_selected_features.pkl"
df = pd.read_pickle(INPUT_FILE)

# 2. Drop records missing either target or text
df_clean = df.dropna(subset=["Combined_Narrative", "Primary Problem"]).copy()

# 3. Restrict to top 4 dominant classes (drops sparse / fringe categories)
TOP_K = 4
top_classes = df_clean["Primary Problem"].value_counts().nlargest(TOP_K).index
df_filtered = df_clean[df_clean["Primary Problem"].isin(top_classes)].copy()

print(f"Total samples after top {TOP_K} class filtering: {len(df_filtered)}")
print("\nClass Counts:")
print(df_filtered["Primary Problem"].value_counts())

X = df_filtered["Combined_Narrative"]
y = df_filtered["Primary Problem"]

# 4. Stratified Train / Test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Lightweight NLP baseline: standard TF-IDF (no lemmatization or stemming)
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000,
    ngram_range=(1, 2),  # unigrams and bigrams
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Linear baseline classifier with class weighting
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# 7. Performance evaluation
y_pred = model.predict(X_test_vec)

print("\n" + "=" * 60)
print("FEASIBILITY BASELINE CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred))

print("\nCONFUSION MATRIX:")
conf_matrix = pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=top_classes),
    index=[f"True: {label}" for label in top_classes],
    columns=[f"Pred: {label}" for label in top_classes],
)
print(conf_matrix)
