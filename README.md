# OCR for wine labels
Project 1 for Applied Computer Vision

### Usage
To run inferencing with models:
1. Download the models
2. Install EasyOCR by `pip install EasyOCR`
3. Place the models in model path, which by default is `~/.EasyOCR/model`
4. Place the corresponding .py and .yaml files in wine_label_config in `~/.EasyOCR/user_networks`
5. (Signup required) Download wine label dataset here https://universe.roboflow.com/wine-label/wine-label-detection/dataset/7
6. Use the scripts in data_processing_scripts to process the data. Remember to modify file names as needed.

    1. Run generate_label.sh. This step requires installing tesseract by following this https://tesseract-ocr.github.io/tessdoc/Installation.html
    2. Run clean_labels.py. This removes the file names of images with no tesseract output from your label file
    3. (Optional) Manually clean up the labels by correcting the labels generated by tesseract
    4. Run get_file_name.py to get a list of file names that are in your labels file
    5. Run clean_dir.py to remove the images that aren't in your labels file from the directory using the output file of the previous step
    6. Run txt_to_csv twice. First for the commented part and second for the uncommented part. After that your labels file should be in csv format
7. Move your data to the location you prefer and modify the path in wine_label_demo.ipynb.
8. Run wine_label_demo.ipynb. The general-purpose model will be automatically downloaded for you after running the second cell.
