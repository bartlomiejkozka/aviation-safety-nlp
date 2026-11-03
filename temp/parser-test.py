import pandas as pd
import numpy as np

FILE_PATH = "ASRS_DBOnline.csv"

print(f"1. Wczytywanie pliku: {FILE_PATH}...")

# header=1 oznacza, że właściwe nazwy kolumn (ACN, Date, Primary Problem) są w 2. wierszu
try:
    df = pd.read_csv(
        FILE_PATH,
        header=1,
        low_memory=False,
        encoding="utf-8"
    )
except UnicodeDecodeError:
    df = pd.read_csv(
        FILE_PATH,
        header=1,
        low_memory=False,
        encoding="latin1"
    )

print(f"Wczytano: {df.shape[0]} wierszy i {df.shape[1]} kolumn.\n")

# Wszystkie kolumny (nazwy z 2. wiersza; pierwszy wiersz to tylko grupa/typ)
print("Wszystkie kolumny:")
for i, col in enumerate(df.columns):
    print(f"{i:3d}. {col}")
print()

# --- 2. WERYFIKACJA TARGETU (Primary Problem) ---
print("=" * 60)
print("1. STATYSTYKI ZMIENNEJ DOCELOWEJ (Primary Problem)")
print("=" * 60)

if "Primary Problem" in df.columns:
    target_counts = df["Primary Problem"].value_counts(dropna=False)
    target_pct = df["Primary Problem"].value_counts(dropna=False, normalize=True) * 100
    
    target_stat = pd.DataFrame({
        "Liczba rekordów": target_counts,
        "Udział %": target_pct.round(2)
    })
    print(target_stat)
    print(f"\nUnikalnych klas: {df['Primary Problem'].nunique(dropna=True)}")
    print(f"Liczba braków danych (NaN): {df['Primary Problem'].isna().sum()}")
else:
    print("Nie znaleziono kolumny 'Primary Problem'. Dostępne kolumny:")
    print(list(df.columns))

# --- 3. ANALIZA CECH TEKSTOWYCH (Narrative & Synopsis) ---
print("\n" + "=" * 60)
print("2. ANALIZA KOLUMN TEKSTOWYCH (X - Features)")
print("=" * 60)

# Pandas automatycznie nazwie drugą narrację jako 'Narrative.1'
narr_cols = [c for c in df.columns if c.startswith("Narrative")]
print(f"Wykryte kolumny z relacjami pilotów: {narr_cols}")

# Łączymy narracje, jeśli dla danego incydentu wypowiedziało się dwóch członków załogi
def combine_narratives(row):
    texts = [str(row[c]) for c in narr_cols if pd.notna(row[c]) and str(row[c]).strip() != ""]
    return " ".join(texts) if texts else np.nan

df["full_narrative"] = df.apply(combine_narratives, axis=1)

# Statystyki długości tekstów
df["text_word_count"] = df["full_narrative"].dropna().apply(lambda x: len(x.split()))
df["text_char_count"] = df["full_narrative"].dropna().apply(lambda x: len(x))

print("\nRozkład długości narracji (liczba słów na jeden raport):")
print(df["text_word_count"].describe().round(1))

# --- 4. PODSUMOWANIE JAKOŚCI DANYCH POD KLASYFIKACJĘ ---
print("\n" + "=" * 60)
print("3. JAKOŚĆ ZBIORU POD TRENING NLP (Gotowe pary X -> Y)")
print("=" * 60)

valid_pairs = df[df["Primary Problem"].notna() & df["full_narrative"].notna()]
print(f"Liczba rekordów posiadających ZARÓWNO tekst jak i etykietę: {len(valid_pairs)} / {len(df)}")

# Symulacja top 4 klas
top_4_classes = valid_pairs["Primary Problem"].value_counts().nlargest(4).index.tolist()
df_top4 = valid_pairs[valid_pairs["Primary Problem"].isin(top_4_classes)]

print(f"\nJeśli ograniczymy się do 4 dominujących klas:")
print(f"Wybrane klasy: {top_4_classes}")
print(f"Zostaje rekordów do modelowania: {len(df_top4)} ({round(len(df_top4)/len(valid_pairs)*100, 1)}% całego zbioru)")
print("\nRozkład w top 4 klasach:")
print(df_top4["Primary Problem"].value_counts())
