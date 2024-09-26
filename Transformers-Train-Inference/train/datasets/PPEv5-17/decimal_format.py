import json

def format_numbers(data):
    # Função para formatar números com uma casa decimal
    def format_number(n):
        return float(f"{n:.1f}")

    # Iterar sobre cada item no dataset
    for item in data:
        # Formatar números em bbox
        item['objects']['bbox'] = [[format_number(coord) for coord in bbox] for bbox in item['objects']['bbox']]
        # Formatar números em area
        item['objects']['area'] = [format_number(area) for area in item['objects']['area']]
    
    return data

# Carregar o arquivo JSON
with open('validation/metadata.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Formatar os números
formatted_data = format_numbers(data)

# Salvar o arquivo JSON formatado
with open('metadata.jsonl', 'w') as f:
    for item in formatted_data:
        f.write(json.dumps(item) + '\n')
