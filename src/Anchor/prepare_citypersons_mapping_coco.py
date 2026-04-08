from glob import glob
import os
import json
from tqdm import tqdm
import copy


cityscapes_dataset_path = os.environ['CITYSCAPES_DATASET']
ano_source_folder = os.path.join(cityscapes_dataset_path, 'gtBboxCityPersons/val') #Annotation source folder
output_folder = os.path.join(cityscapes_dataset_path, 'gtBboxCityPersons')

img_suffix = "leftImg8bit.png"

# label -> [category_id, isCrowd]
# None to discard the category
lbl_map = {
    'ignore': None,#[91, 0],
    'pedestrian' : [1, 0],
    'rider' : [1, 0],
    'sitting person': [1, 0],
    'person (other)': [1, 0],
    'person group': [1, 1]
}  # ignore,ped,rider,sit,other,group

seq_folders = [os.path.basename(os.path.normpath(path)) for path in glob(os.path.join(ano_source_folder, "*/"))]

ann_jsons = glob(os.path.join(ano_source_folder,"*","*.json"))

cat_str = '[{"supercategory": "person","id": 1,"name": "person"},{"supercategory": "vehicle","id": 2,"name": "bicycle"},{"supercategory": "vehicle","id": 3,"name": "car"},{"supercategory": "vehicle","id": 4,"name": "motorcycle"},{"supercategory": "vehicle","id": 5,"name": "airplane"},{"supercategory": "vehicle","id": 6,"name": "bus"},{"supercategory": "vehicle","id": 7,"name": "train"},{"supercategory": "vehicle","id": 8,"name": "truck"},{"supercategory": "vehicle","id": 9,"name": "boat"},{"supercategory": "outdoor","id": 10,"name": "traffic light"},{"supercategory": "outdoor","id": 11,"name": "fire hydrant"},{"supercategory": "outdoor","id": 13,"name": "stop sign"},{"supercategory": "outdoor","id": 14,"name": "parking meter"},{"supercategory": "outdoor","id": 15,"name": "bench"},{"supercategory": "animal","id": 16,"name": "bird"},{"supercategory": "animal","id": 17,"name": "cat"},{"supercategory": "animal","id": 18,"name": "dog"},{"supercategory": "animal","id": 19,"name": "horse"},{"supercategory": "animal","id": 20,"name": "sheep"},{"supercategory": "animal","id": 21,"name": "cow"},{"supercategory": "animal","id": 22,"name": "elephant"},{"supercategory": "animal","id": 23,"name": "bear"},{"supercategory": "animal","id": 24,"name": "zebra"},{"supercategory": "animal","id": 25,"name": "giraffe"},{"supercategory": "accessory","id": 27,"name": "backpack"},{"supercategory": "accessory","id": 28,"name": "umbrella"},{"supercategory": "accessory","id": 31,"name": "handbag"},{"supercategory": "accessory","id": 32,"name": "tie"},{"supercategory": "accessory","id": 33,"name": "suitcase"},{"supercategory": "sports","id": 34,"name": "frisbee"},{"supercategory": "sports","id": 35,"name": "skis"},{"supercategory": "sports","id": 36,"name": "snowboard"},{"supercategory": "sports","id": 37,"name": "sports ball"},{"supercategory": "sports","id": 38,"name": "kite"},{"supercategory": "sports","id": 39,"name": "baseball bat"},{"supercategory": "sports","id": 40,"name": "baseball glove"},{"supercategory": "sports","id": 41,"name": "skateboard"},{"supercategory": "sports","id": 42,"name": "surfboard"},{"supercategory": "sports","id": 43,"name": "tennis racket"},{"supercategory": "kitchen","id": 44,"name": "bottle"},{"supercategory": "kitchen","id": 46,"name": "wine glass"},{"supercategory": "kitchen","id": 47,"name": "cup"},{"supercategory": "kitchen","id": 48,"name": "fork"},{"supercategory": "kitchen","id": 49,"name": "knife"},{"supercategory": "kitchen","id": 50,"name": "spoon"},{"supercategory": "kitchen","id": 51,"name": "bowl"},{"supercategory": "food","id": 52,"name": "banana"},{"supercategory": "food","id": 53,"name": "apple"},{"supercategory": "food","id": 54,"name": "sandwich"},{"supercategory": "food","id": 55,"name": "orange"},{"supercategory": "food","id": 56,"name": "broccoli"},{"supercategory": "food","id": 57,"name": "carrot"},{"supercategory": "food","id": 58,"name": "hot dog"},{"supercategory": "food","id": 59,"name": "pizza"},{"supercategory": "food","id": 60,"name": "donut"},{"supercategory": "food","id": 61,"name": "cake"},{"supercategory": "furniture","id": 62,"name": "chair"},{"supercategory": "furniture","id": 63,"name": "couch"},{"supercategory": "furniture","id": 64,"name": "potted plant"},{"supercategory": "furniture","id": 65,"name": "bed"},{"supercategory": "furniture","id": 67,"name": "dining table"},{"supercategory": "furniture","id": 70,"name": "toilet"},{"supercategory": "electronic","id": 72,"name": "tv"},{"supercategory": "electronic","id": 73,"name": "laptop"},{"supercategory": "electronic","id": 74,"name": "mouse"},{"supercategory": "electronic","id": 75,"name": "remote"},{"supercategory": "electronic","id": 76,"name": "keyboard"},{"supercategory": "electronic","id": 77,"name": "cell phone"},{"supercategory": "appliance","id": 78,"name": "microwave"},{"supercategory": "appliance","id": 79,"name": "oven"},{"supercategory": "appliance","id": 80,"name": "toaster"},{"supercategory": "appliance","id": 81,"name": "sink"},{"supercategory": "appliance","id": 82,"name": "refrigerator"},{"supercategory": "indoor","id": 84,"name": "book"},{"supercategory": "indoor","id": 85,"name": "clock"},{"supercategory": "indoor","id": 86,"name": "vase"},{"supercategory": "indoor","id": 87,"name": "scissors"},{"supercategory": "indoor","id": 88,"name": "teddy bear"},{"supercategory": "indoor","id": 89,"name": "hair drier"},{"supercategory": "indoor","id": 90,"name": "toothbrush"}]'#,{"supercategory": "person","id": 91,"name": "ignore"}]'

cat = json.loads(cat_str)
json_dict = {"images":[], "annotations": [], "categories": cat}
image_id = 1
ann_id = 1

for jsonFile in tqdm(ann_jsons):
    data = json.load(open(jsonFile, 'r'))
    fn = os.path.basename(os.path.normpath(jsonFile))

    # Replace the trailing _*.json by the the suffix of img files
    folder_name = fn.split("_")[0]
    img_name = fn.replace(fn.split("_")[-1], img_suffix)
    height = data["imgHeight"]
    width = data["imgWidth"]
    image = {'file_name': os.path.join(folder_name,img_name), 'height': height, 'width': width,
                 'id':image_id}
    for bbox_ann in data['objects']:
        bbox = bbox_ann['bbox']
        area = bbox[-1] * bbox[-2]
        if lbl_map[bbox_ann['label']]:

            cat_id, isCrowd = lbl_map[bbox_ann['label']]
            ann = {'area': area, 'iscrowd': isCrowd, 'image_id':
                    image_id, 'bbox': bbox,
                    'category_id': cat_id, 'id': ann_id,
                    'segmentation': []}
            json_dict['annotations'].append(ann)
            ann_id += 1
        else:
            continue

    image_id += 1

    json_dict["images"].append(image)

print("Objects count {}".format(ann_id - 1))

# Nokia Code
# Resize the the annotation, if needed
#for res in [25,50,75,100]:
#    if res != 100:
#        ratio = res/100
#        saved_dict = copy.deepcopy(json_dict)
#        for img in saved_dict["images"]:
#            img["width"] = int(img["width"] * ratio)
#            img["height"] = int(img["height"] * ratio)
#        for ann in saved_dict["annotations"]:
#            for i in range(len(ann["bbox"])):
#                ann["bbox"][i] = int(ann["bbox"][i] * ratio)
#    else:
#        saved_dict = json_dict
#    result_json_file = os.path.join(output_folder, f"cityperson_object_det_val_{res}.json")
#    json_str = json.dumps(saved_dict)

result_json_file = os.path.join(output_folder, f"citypersons_object_det_val.json")
json_fp = open(result_json_file, 'w')
json_str = json.dumps(json_dict, indent='\t')
json_fp.write(json_str)
print("Saved the annotation file to {}".format(result_json_file))
json_fp.close()

