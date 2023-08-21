# Pathology Labeling Tool
This project aims to assist pathologists to label pathological images in a more efficient way. By using neural networks to pre-classify images, pathologists just need to re-label misclassified tissues, thus improving the efficiency of labeling. Through the web interface, users can upload files, label data, and view and download labeled results.

<img src="https://github.com/v1x99y7/pathology_labeling_tool/blob/main/figures/labeling.png" width = "500px"/>

## Examples
### 1. Upload files
#### Classify the uploaded files and insert the result into the database.
* Homepage
<img src="https://github.com/v1x99y7/pathology_labeling_tool/blob/main/figures/index.png" width = "500px"/>

* Click the select files button, select uploaded files, and click the upload button.
<img src="https://github.com/v1x99y7/pathology_labeling_tool/blob/main/figures/upload1.png" width = "500px"/>

* Classification result
<img src="https://github.com/v1x99y7/pathology_labeling_tool/blob/main/figures/upload2.png" width = "500px"/>

### 2. Label data
#### Re-label the misclassified tissues to update the label in the database.
* Click the tissue button (ex: NORM) to select the predicted tissue type you want to label.
<img src="https://github.com/v1x99y7/pathology_labeling_tool/blob/main/figures/labeling1.png" width = "500px"/>

* Click the misclassified tissue and re-label it by clicking the popup tissue button.
<img src="https://github.com/v1x99y7/pathology_labeling_tool/blob/main/figures/labeling2.png" width = "500px"/>

* Click the confirm button to update the label.
<img src="https://github.com/v1x99y7/pathology_labeling_tool/blob/main/figures/labeling3.png" width = "500px"/>

### 3. View and download labeled results.
#### Select data from the database.
* Click the result button on the homepage, and used the drop-down list to search the data for the selected condition.
<img src="https://github.com/v1x99y7/pathology_labeling_tool/blob/main/figures/result1.png" width = "500px"/>

* Click the download button to download the result.csv file.
<img src="https://github.com/v1x99y7/pathology_labeling_tool/blob/main/figures/result2.png" width = "500px"/>

* result.csv file
<img src="https://github.com/v1x99y7/pathology_labeling_tool/blob/main/figures/result3.png" width = "500px"/>

## Requirement
* Flask, Pytorch, MySQL

## Note
* ADI: Adipose
* BACK: background
* DEB: debris
* LYM: lymphocytes
* MUC: mucus
* MUS: smooth muscle
* NORM: normal colon mucosa
* STR: cancer-associated stroma
* TUM: colorectal adenocarcinoma epithelium
