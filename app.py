from flask import Flask, render_template, request
import db, classifier
from PIL import Image
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<tissue>', methods=['GET', 'POST'])
def label_patches(tissue):
    if request.method == 'POST':
        length = int(request.values.get('length'))
        for i in range(1, length+1):
            sql = f"""
            UPDATE `patch`
            SET `true_label` = '{request.values.get(f"label{i}")}', `verification` = 'verified'
            WHERE `patch_id` = '{request.values.get(f"id{i}")}';
            """
            db.modify_data(sql)

    sql = f"""
    SELECT `patch_id`, `patch_path`
    FROM `patch`
    WHERE `verification` = 'unverified' AND `pred_label` = '{tissue}'
    LIMIT 12;
    """
    datas = db.query_data(sql)
    return render_template('label_patches.html', tissue=tissue, datas=datas)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    filenames = []
    paths = []
    pred_labels = []

    if request.method == 'POST':
        uploaded_files = request.files.getlist("file[]")
        for file in uploaded_files:
            # save
            image = Image.open(file)
            filename = f'{file.filename.split(".")[0]}.png'
            filenames.append(filename)
            path = f'static/data/{filename}'
            image.save(path)
            paths.append(path)

            # classify
            pred_label = classifier.classify(path)
            pred_labels.append(pred_label)

            # insert data
            sql = f"""
            INSERT INTO `patch`(`filename`, `true_label`, `pred_label`, `patch_path`) VALUES('{filename}', '{pred_label}', '{pred_label}', '{path}');
            """
            db.insert_data(sql)
    
    result = {'ADI':[], 'BACK':[], 'DEB':[], 'LYM':[], 'MUC':[], 'MUS':[], 'NORM':[], 'STR':[], 'TUM':[]}
    file_number = len(filenames)
    for i in range(file_number):
        result[pred_labels[i]].append(paths[i])
    
    return render_template('upload.html', file_number=file_number, result=result)

@app.route('/result', methods=['GET', 'POST'])
def result():
    sql = f"""
    SELECT `patch_id`, `filename`, `verification`, `true_label`, `pred_label`, `patch_path`
    FROM `patch`;
    """
    datas = db.query_data(sql)
    
    columns = datas[0].keys()
    dataList = []
    for data in datas:
        dataList.append(data.values())
    df = pd.DataFrame(dataList, columns=columns)
    df.to_csv('static/result/result.csv', index=False)

    if request.method == 'POST':
        sql = f"""
        SELECT `patch_id`, `filename`, `verification`, `true_label`, `pred_label`, `patch_path`
        FROM `patch`
        WHERE `verification` LIKE '{request.values.get('verification')}' AND `true_label` LIKE '{request.values.get('true_label')}' AND `pred_label` LIKE '{request.values.get('pred_label')}';
        """
        datas = db.query_data(sql)
        
    return render_template('result.html', datas=datas)

if __name__ == '__main__':
    app.run()