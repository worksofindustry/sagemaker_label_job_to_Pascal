import json
import s3fs
import pandas as pd
import os.path
from jinja2 import Environment, PackageLoader

class Writer:
    def __init__(self, path, width, height, depth=3, database='Unknown', segmented=0):
        environment = Environment(loader=PackageLoader('pascal_voc_writer', 'templates'), keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': path,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    def addObject(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    def save(self, annotation_path):
        with open(annotation_path, 'w') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)

def parse_gt_output(manifest_path, job_name):
    """
    Captures the json GroundTruth bounding box annotations into a pandas dataframe

    Input:
    manifest_path: S3 path to the annotation file
    job_name: name of the GroundTruth job

    Returns:
    df_bbox: pandas dataframe with bounding box coordinates
             for each item in every image
    """
    filesys = s3fs.S3FileSystem()
    with filesys.open(manifest_path) as fin:
        annot_list = []
        for line in fin.readlines():
            record = json.loads(line)
            if job_name in record.keys():  # is it necessary?
                image_file_path = record["source-ref"]
                image_file_name = image_file_path.split("/")[-1]
                class_maps = record[f"{job_name}-metadata"]["class-map"]

                imsize_list = record[job_name]["image_size"]
                assert len(imsize_list) == 1
                image_width = imsize_list[0]["width"]
                image_height = imsize_list[0]["height"]
                image_depth = imsize_list[0]["depth"]

                for annot in record[job_name]["annotations"]:
                    left = annot["left"]
                    top = annot["top"]
                    height = annot["height"]
                    width = annot["width"]
                    class_name = class_maps[f'{annot["class_id"]}']

                    annot_list.append(
                        [
                            image_file_name,
                            class_name,
                            left,
                            top,
                            height,
                            width,
                            image_width,
                            image_height,
                            image_depth
                        ]
                    )

        df_bbox = pd.DataFrame(
            annot_list,
            columns=[
                "img_file",
                "category",
                "box_left",
                "box_top",
                "box_height",
                "box_width",
                "img_width",
                "img_height",
                "image_depth"
            ],
        )

    return df_bbox    


def get_cats(json_file):
    """
    Makes a list of the category names in proper order

    Input:
    json_file: s3 path of the json file containing the category information

    Returns:
    cats: List of category names
    """
    filesys = s3fs.S3FileSystem()
    with filesys.open(json_file) as fin:
        line = fin.readline()
        record = json.loads(line)
        labels = [item["label"] for item in record["labels"]]

    return labels


def split_to_train_test(df, label_column, train_frac=0.8):
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    labels = df[label_column].unique()
    for lbl in labels:
        lbl_df = df[df[label_column] == lbl]
        lbl_train_df = lbl_df.sample(frac=train_frac)
        lbl_test_df = lbl_df.drop(lbl_train_df.index)
        print ('\n%s:\n---------\ntotal:%d\ntrain_df:%d\ntest_df:%d' % (lbl, len(lbl_df), len(lbl_train_df), len(lbl_test_df)))
        train_df = train_df.append(lbl_train_df)
        test_df = test_df.append(lbl_test_df)

    return train_df, test_df



def main():
    # Get Annotation Job Configurations
    with open("input.json") as fjson:
        input_dict = json.load(fjson)
    
    s3_bucket = input_dict["s3_bucket"]
    job_id = input_dict["job_id"]
    gt_job_name = input_dict["ground_truth_job_name"]
    photo_dir = input_dict["photo_dir"]
    manifest_path = f"s3://{s3_bucket}/{job_id}/ground_truth_annots/{gt_job_name}/manifests/output/output.manifest"
    images_path = f"s3://{s3_bucket}/{job_id}/{photo_dir}/"
    
    
    df_annot = parse_gt_output(manifest_path, gt_job_name)
    s3_path_cats = (f"s3://{s3_bucket}/{job_id}/ground_truth_annots/{gt_job_name}/annotation-tool/data.json")
    categories = get_cats(s3_path_cats)
    df_annot["image_folder"] = images_path
    df_annot["full_file_path"] = images_path + df_annot["img_file"]
    
    #Convert DataFrame to PascalVoc format (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
    df_annot["x_1"] = df_annot["box_left"]
    df_annot["y_1"] = df_annot["box_top"] + df_annot["box_height"]
    df_annot["x_2"] = df_annot["box_left"] + df_annot["box_width"]
    df_annot["y_2"] = df_annot["box_top"]
    
    
    
    for i, row in df_annot.iterrows():
        # Writer(path, width, height)
        writer = Writer(f"{row['full_file_path']}", f"{row['img_width']}", f"{row['img_height']}")
        # ::addObject(name, xmin, ymin, xmax, ymax)
        writer.addObject(f"{row['category']}", f"{row['x_1']}", f"{row['y_2']}", f"{row['x_2']}", f"{row['y_1']}")
        # ::save(path)
        save_to = './Annotations/' + f"{row['category']}" + "_" + str(i) + "_" + f"{row['img_file']}" + '.xml'
        writer.save(save_to)
        
    # Generate Train/Test Image Sets
    train, test = split_to_train_test(df_annot[["category","img_file"]], "category", train_frac=0.85)    
    
    twd = os.getcwd() + '/Train_Split'
    
    for c in categories:
        #Train Set
        t = train.loc[train['category']==c]
        base_filename = c+'_train.txt'
        with open(os.path.join(twd, base_filename),'w') as f:
            t['img_file'].to_csv(f, index=False, header = False, encoding='utf-8', line_terminator='\n')   
        #Validation Set
        val = test.loc[test['category']==c]
        base_filename = c+'_val.txt'
        with open(os.path.join(twd, base_filename),'w') as f:
            val['img_file'].to_csv(f, index=False, header = False, encoding='utf-8', line_terminator='\n')           


if __name__ == "__main__":
    main()

