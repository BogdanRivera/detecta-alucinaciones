import requests
from sentence_transformers import SentenceTransformer, util
import json 
import numpy as np 


model = SentenceTransformer('all-MiniLM-L6-v2')

def get_wikipedia_article(title, lang="es"):
    """
    Obtiene el texto completo de un artículo de Wikipedia.

    :param title: Título del artículo.
    :param lang: Idioma (por defecto 'es' para español).
    :return: Texto del artículo.
    """
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "format": "json",
        "explaintext": True,  # Extraer texto plano
    }
    response = requests.get(url, params=params)
    pages = response.json().get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id != "-1":  # Página encontrada
            return page.get("extract", "No extract available.")
    return "No extract available."

def extract_train_data_es(rute_data = './train/mushroom.es-train_nolabel.v1.jsonl'): 
    data = []
    with open(rute_data, 'r') as j:
        data = [json.loads(line) for line in j]
    return data

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


if __name__ == '__main__':
    data = extract_train_data_es()
    print(data[0]['model_input'])


    #article_title = "Inteligencia artificial"
    #article_text = get_wikipedia_article(article_title)
    #print(article_text)
