import json

# Função para converter bbox do formato COCO para o formato desejado
def convert_bbox_coco_to_custom(bbox):
    x, y, width, height = bbox
    return [x, y, width, height]

# Carregar o arquivo COCO JSON
with open('_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

# Dicionário para mapear categorias
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

# Dicionário para armazenar as anotações por imagem
annotations_by_image = {}
for ann in coco_data['annotations']:
    image_id = ann['image_id']
    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = {
            "bbox": [],
            "categories": [],
            "id": [],
            "area": []
        }
    annotations_by_image[image_id]["bbox"].append(convert_bbox_coco_to_custom(ann['bbox']))
    annotations_by_image[image_id]["categories"].append(ann['category_id'])
    annotations_by_image[image_id]["id"].append(ann['id'])
    annotations_by_image[image_id]["area"].append(ann['area'])

# Criar o arquivo metadata.jsonl
with open('metadata.jsonl', 'w') as f:
    for img in coco_data['images']:
        image_id = img['id']
        file_name = img['file_name']
        if image_id in annotations_by_image:
            objects = annotations_by_image[image_id]
            metadata = {
                "file_name": file_name,
                "objects": objects
            }
            f.write(json.dumps(metadata) + '\n')
