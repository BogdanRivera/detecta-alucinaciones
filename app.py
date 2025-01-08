import wikipedia
from transformers import pipeline
import spacy
from sentence_transformers import CrossEncoder
import numpy as np

# Carga de modelo 
similarity_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
nlp = spacy.load("en_core_web_sm")

# Extracción de afirmaciones 
def extract_claims(text):
    """Extrae afirmaciones clave del texto."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

# Búsqueda en Wikipedia
def search_wikipedia(query):
    """Busca evidencia en Wikipedia."""
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation error: {e.options[:3]}"
    except wikipedia.exceptions.PageError:
        return "No se encontró una página relevante."

# Validación de afirmaciones
def validate_claims(claims):
    """Valida las afirmaciones comparándolas con evidencia recuperada."""
    results = []
    for claim in claims:
        evidence = search_wikipedia(claim)
        if "Disambiguation error" in evidence or "No se encontró" in evidence:
            results.append({
                "claim": claim,
                "evidence": evidence,
                "result": "Sin evidencia clara"
            })
            continue

        # Comparar similitud semántica
        try:
            scores = similarity_model.predict([(claim, evidence)])
            score = scores[0]
            if isinstance(score, np.ndarray):  # Verifica si score es un array de NumPy
                score = score.item()
            label = "Consistente" if score > 0.5 else "Inconsistente"
        except Exception as e:
            score = 0.0
            label = f"Error: {str(e)}"

        results.append({
            "claim": claim,
            "evidence": evidence,
            "result": label,
            "score": score  # Mantén el score como un valor flotante
        })
    return results

# Resultados
def generate_report(results):
    for result in results:
        print(f"Afirmación: {result['claim']}")
        print(f"Evidencia: {result['evidence']}")
        print(f"Resultado: {result['result']}, Confianza: {result.get('score', 'N/A')}\n")

# Ejemplo del texto
generated_text = "Although their name may suggest otherwise, whales are marine mammals. The blue whale can reach a length of up to 30 metres and weigh more than 180 tonnes."

# Ejecución del pipeline 
afirmaciones = extract_claims(generated_text)
resultados = validate_claims(afirmaciones)
generate_report(resultados)
