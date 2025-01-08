import requests
from sentence_transformers import SentenceTransformer, util


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

def calculate_similarity(response, documents):
    """
    Calcula la similitud máxima entre una respuesta y documentos recuperados.
    :param response: Respuesta generada por el modelo.
    :param documents: Lista de documentos recuperados 
    :return: La similitud máxima encontrada.
    """
    response_embedding = model.encode(response, convert_to_tensor=True)
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(response_embedding, document_embeddings)
    return max(similarities).item()

def classify_hallucination(similarity_score, threshold=0.1):
    """
    Clasifica una respuesta como alucinación o no basada en el puntaje de similitud.
    :param similarity_score: Puntaje de similitud máxima.
    :param threshold: Umbral para decidir si es factualmente correcto.
    :return: False si es factualmente correcto, True si es alucinación.
    """
    return similarity_score >= threshold

def verify_facts_with_wikipedia(query, model_output):
    """
    Verifica la respuesta generada comparándola con información recuperada de Wikipedia.
    :param query: Pregunta del usuario.
    :param model_output: Respuesta generada por el modelo.
    :return: Diccionario con el puntaje de similitud y la etiqueta de alucinación.
    """
    # Recuperar fragmentos de Wikipedia
    snippets = get_wikipedia_article(query)
    if not snippets:
        return {"similarity_score": 0, "hallucination": True}  # Sin información

    # Calcular similitud
    similarity_score = calculate_similarity(model_output, snippets)

    # Clasificar como alucinación
    is_factually_correct = classify_hallucination(similarity_score)
    return {"similarity_score": similarity_score, "hallucination": not is_factually_correct}




#Código con palabras clave 
import spacy

nlp = spacy.load("es_core_news_sm")

def extract_entities(text):
    doc = nlp(text)
    return {ent.text for ent in doc.ents}

def check_key_entities(response, document):
    response_entities = extract_entities(response)
    document_entities = extract_entities(document)
    return response_entities.issubset(document_entities)

def validate_with_keywords(response, documents):
    for doc in documents:
        if check_key_entities(response, doc):
            return True  # Entidades clave coinciden
    return False

def verify_facts_with_keywords(query, model_output):
    snippets = get_wikipedia_article(query)
    if not snippets:
        return {"similarity_score": 0, "hallucination": True}

    similarity_score = calculate_similarity(model_output, snippets)
    is_semantically_correct = classify_hallucination(similarity_score)

    # Validar entidades clave
    is_keyword_correct = validate_with_keywords(model_output, [snippets])

    return {
        "similarity_score": similarity_score,
        "hallucination": not (is_semantically_correct and is_keyword_correct),
    }
#Fin con palabras clave


query = "¿Cuándo fue creada la comuna suiza Gampel-Bratsch?"
model_output = "La comuna suiza Gampel-Bratsch fue creada el 1 de enero de 2004. Esta nueva comuna surgió de la fusión de dos antiguas comunas: Gampel y Bratsch. Los habitantes de ambas comunas votaron a favor de esta unión el 20 de enero de 2008."
result = verify_facts_with_keywords(query, model_output)
print(result)







#article_title = "Inteligencia artificial"
#article_text = get_wikipedia_article(article_title)
#print(article_text)
