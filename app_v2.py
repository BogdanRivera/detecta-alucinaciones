from sentence_transformers import CrossEncoder
import spacy
import wikipedia

# Carga de modelo
similarity_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
nlp = spacy.load("en_core_web_sm")

# Extracción de afirmaciones
def extract_claims(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


def search_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=5)
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

# Datos de prueba 
claims = ["La Tierra es plana", "El agua hierve a 100 grados Celsius"] 
# Validación de afirmaciones 
results = validate_claims(claims)
# Impresión de resultados
for result in results: 
    print(result)





#Usar la api de wikipedia
# Prueba de la función con una consulta de ejemplo
#resultado = search_wikipedia("Mexico")
#print(resultado)

# Usar extracción de afirmaciones 
# text = "Bogdan loves programming. He believes it's the future. Artificial Intelligence is fascinating."

# # Extraer afirmaciones del texto
# claims = extract_claims(text)

# print("Afirmaciones extraídas:")
# for claim in claims:
#     print(claim)

# # Crear pares de afirmaciones para calcular similitudes
# pairs = [(claim1, claim2) for i, claim1 in enumerate(claims) for j, claim2 in enumerate(claims) if i < j]


# similarities = similarity_model.predict(pairs)

# # Mostrar las similitudes calculadas
# print("\nSimilitudes de las frases:")
# for i, (pair, similarity) in enumerate(zip(pairs, similarities)):
#     print(f"Pares {i+1}: {pair}")
#     print(f"Similitud: {similarity}\n")
