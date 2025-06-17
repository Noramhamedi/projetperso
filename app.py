from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import re
import string
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Set
import matplotlib.pyplot as plt
import pickle
import time
from pathlib import Path
from gensim.models import KeyedVectors
from itertools import product
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'W3amN0ur3nD!1278'  # Clé secrète Flask

# === CONFIGURATION ===
MODEL_PATH = "wiki-news-300d-1M-subword.vec"
CACHE_DIR = "./vector_cache"
TOP_N = 10
MIN_SIMILARITY = 0.6

Path(CACHE_DIR).mkdir(exist_ok=True, parents=True)

# === FONCTIONS SEMANTIQUES ===
def load_cached_model():
    cache_file = f"{CACHE_DIR}/model_cache.pkl"
    if os.path.exists(cache_file):
        print("Chargement depuis le cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        print("Chargement initial (peut prendre 1-2 minutes)...")
        start = time.time()
        model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False, unicode_errors='ignore')
        with open(cache_file, 'wb') as f:
            pickle.dump(model, f, protocol=4)
        print(f"Modèle chargé en {time.time()-start:.2f}s")
        return model

def trouver_synonymes(mot: str, model) -> List[str]:
    try:
        similar_words = model.most_similar(mot, topn=TOP_N*2)
        filtered = [
            word for word, score in similar_words
            if score > MIN_SIMILARITY and word.lower() != mot.lower()
        ][:TOP_N]
        return filtered
    except KeyError:
        print(f"Mot inconnu dans le modèle : {mot}")
        return []

model = load_cached_model()

# === ANALYSE DE DONNEES ===
def tokenize_text(text: Union[str, float]) -> List[str]:
    if pd.isna(text):
        return []
    text = re.sub(r'\d+', '', str(text).lower())
    text_clean = re.sub(r'[^\w\s]', ' ', text)
    return text_clean.split()

def filtrer_mots(liste_mots: List[str], mots_vides_supplementaires: List[str] = None) -> List[str]:
    mots_vides = {
        'le', 'la', 'les', 'de', 'des', 'un', 'une', 'et', 'ou', 'dans',
        'pour', 'au', 'aux', 'avec', 'sur', 'par', 'est', 'son', 'sa', 'ses','en', 'du',
        'ces', 'cet', 'cette', 'ceux', 'bonjour', 'merci', 'salut', 'a', 'à', 'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
        'd', 'avons', 'cordialement', 'que', 'l', 'votre', 'pas', 'n', 'm', 'ce', 'vos', 'notre', 'ne', 'plus', 'qui','après', 'afin', 'car', 's','via', 'si', 'donc',
        'sans','y','mais', 'ai', 'non', 'ont','j', 'cela', 'ci', 'mme'
    }
    if mots_vides_supplementaires:
        mots_vides.update(mot.lower() for mot in mots_vides_supplementaires)
    traducteur = str.maketrans('', '', string.punctuation)
    return [
        mot.translate(traducteur).lower()
        for mot in liste_mots
        if mot and mot.translate(traducteur).lower() not in mots_vides
    ]

def analyser_fichiers(fichiers: List[str]) -> Tuple[List[pd.DataFrame], List[Dict[str, List[int]]]]:
    dataframes = []
    dictionnaires = []

    for fichier in fichiers:
        try:
            df = pd.read_excel(fichier, header=None)
            colonnes_excel = [1, 2, 3, 8, 9, 20, 24, 25, 26, 37, 38, 31, 43, 23]
            noms_colonnes = ["B", "C", "D", "I", "J", "U", "Y", "Z", "AA", "AL", "AM", "AF", "AR", "X"]
            df = df.iloc[1:]  # Ignore la première ligne (header custom)
            df = df[colonnes_excel]
            df.columns = noms_colonnes

            # Mise à jour des colonnes Y et Z si AL et AM non nuls
            df["Y"] = df.apply(lambda row: row["AL"] if pd.notna(row["AL"]) else row["Y"], axis=1)
            df["Z"] = df.apply(lambda row: row["AM"] if pd.notna(row["AM"]) else row["Z"], axis=1)

            colonnes_tokens = ["AR", "AA", "AF"]
            for col in colonnes_tokens:
                df[col] = df[col].apply(tokenize_text)

            colonnes_texte = ["B", "C", "D", "I", "J", "U", "Y", "Z", "AL", "AM"]
            dico_mots = defaultdict(list)

            for i, row in df.iterrows():
                ligne_num = i + 2
                for col in colonnes_tokens:
                    mots = filtrer_mots(row[col])
                    for mot in mots:
                        dico_mots[mot].append(ligne_num)
                for col in colonnes_texte:
                    valeur = row[col]
                    if pd.notna(valeur):
                        dico_mots[str(valeur)].append(ligne_num)

            dataframes.append(df)
            dictionnaires.append(dico_mots)

        except Exception as e:
            print(f"Erreur avec {fichier} : {e}")
            dataframes.append(pd.DataFrame())
            dictionnaires.append(defaultdict(list))

    return dataframes, dictionnaires

def cache_analyse_fichier(fichier: str):
    cache_path = Path(CACHE_DIR) / f"{Path(fichier).stem}_analyse.pkl"
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    df, dico = analyser_fichiers([fichier])
    with open(cache_path, 'wb') as f:
        pickle.dump((df[0], dico[0]), f)
    return df[0], dico[0]

def create_graph(liste_indices, df, col='X', ligne_1_base=True):
    if ligne_1_base:
        liste_indices = [i - 1 for i in liste_indices]

    df = df.copy()
    df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors='coerce')

    df_reclam = df.iloc[liste_indices].copy()

    if df_reclam.empty:
        print("Aucune réclamation trouvée aux indices donnés.")
        return None

    min_date = df_reclam[col].min()
    max_date = df_reclam[col].max()

    debut = pd.Timestamp(f"01/01/{min_date.year}")
    fin = pd.Timestamp(f"31/12/{max_date.year}")

    tous_les_mois = pd.period_range(start=debut, end=fin, freq='M')

    df_reclam['MoisAnnee'] = df_reclam[col].dt.to_period("M")
    counts = df_reclam['MoisAnnee'].value_counts().sort_index()

    df_complet = pd.DataFrame({'MoisAnnee': tous_les_mois})
    counts_df = pd.DataFrame({'MoisAnnee': counts.index, 'Nombre': counts.values})

    df_complet = df_complet.merge(counts_df, how='left', on='MoisAnnee').fillna(0)
    df_complet['Nombre'] = df_complet['Nombre'].astype(int)
    df_complet['MoisAffichage'] = df_complet['MoisAnnee'].dt.strftime("%b %Y")

    plt.figure(figsize=(14, 6))
    bars = plt.bar(df_complet['MoisAffichage'], df_complet['Nombre'], color='skyblue')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')

    plt.title("Nombre de réclamations par mois", pad=20)
    plt.xlabel("Mois", labelpad=10)
    plt.ylabel("Nombre de réclamations", labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.margins(x=0.01)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, df_complet['Nombre'].max() + 1 if df_complet['Nombre'].max() > 0 else 1)
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode the image as base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

def executer_recherche(texte_recherche: str, fichiers: List[str]) -> Tuple[List[Dict], Dict]:
    texte_recherche = texte_recherche.lower()
    liste_mots = filtrer_mots(tokenize_text(texte_recherche))
    print("Recherche : ", texte_recherche)
    print("Mots clés filtrés : ", liste_mots)

    df_list = []
    dict_list = []
    for f in fichiers:
        df_f, dict_f = cache_analyse_fichier(f)
        df_list.append(df_f)
        dict_list.append(dict_f)

    mots_cles_synonymes = []
    for mot in liste_mots:
        synonymes = trouver_synonymes(mot, model)
        mots_cles_synonymes.extend(synonymes)
    mots_cles_synonymes = list(set(mots_cles_synonymes))

    resultats = {}
    for i, dico_mots in enumerate(dict_list):
        lignes_trouvees: Set[int] = set()
        for mot in liste_mots + mots_cles_synonymes:
            lignes_trouvees.update(dico_mots.get(mot, []))
        resultats[i] = {
            "nom_fichier": Path(fichiers[i]).name,
            "lignes": sorted(lignes_trouvees),
            "mots_cles": liste_mots,
            "synonymes": mots_cles_synonymes
        }
        print(f"Fichier {fichiers[i]} - {len(lignes_trouvees)} résultats trouvés")

    return df_list, resultats

# === ROUTES FLASK ===
@app.route('/')
def index():
    return render_template('result1.html', mots_initials=[])

@app.route('/recherche', methods=['POST'])
def route_recherche():
    texte = request.form.get('texte', '')
    fichiers = request.form.getlist('fichiers')
    
    if not fichiers:
        return jsonify({"error": "Aucun fichier sélectionné"}), 400
    
    df_list, resultats = executer_recherche(texte, fichiers)
    
    # Generate graph for the first file's results
    graph_image = None
    if resultats and resultats[0]['lignes']:
        graph_image = create_graph(resultats[0]['lignes'], df_list[0])
    
    response = {
        "summary": {
            "total_results": sum(len(r['lignes']) for r in resultats.values()),
            "by_file": {resultats[k]['nom_fichier']: len(resultats[k]['lignes']) for k in resultats}
        },
        "mots_cles": resultats[0]['mots_cles'] if resultats else [],
        "synonymes": resultats[0]['synonymes'] if resultats else [],
        "graph": graph_image
    }
    
    return jsonify(response)

@app.route('/graph', methods=['POST'])
def generate_graph():
    data = request.json
    file_index = data.get('file_index', 0)
    lines = data.get('lines', [])
    graph_type = data.get('type', 'month')
    
    # Get the corresponding dataframe
    # You might need to modify this to load the correct dataframe
    df = None  # Implement this based on your needs
    
    if graph_type == 'month':
        graph_image = create_graph(lines, df)
    else:
        # Implement other graph types
        graph_image = None
    
    return jsonify({"graph": graph_image})

if __name__ == "__main__":
    app.run(debug=True)