from flask import Flask, render_template, url_for, request, redirect,send_from_directory
from werkzeug.utils import secure_filename
import os
from script import pdf2img, png2jpg, crop_info, OCRextract, invoice_extract, order_extract, receipt_extract, t_extract, json_receipt, json_invoice, json_sale_order
from PIL import Image
from huggingface_hub import hf_hub_download


import numpy as np
import pandas as pd
import json
import csv
import shutil

app = Flask(__name__)


ALLOWED_EXTENSIONS = set(['pdf','png','jpg'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET',"POST"])

def upload():
    try:
        if os.path.exists("./process"):
            shutil.rmtree("./process")
        else:
            print("process file does not exist")

        if os.path.exists("./input"):
            shutil.rmtree("./input")
        else:
            print("input file does not exist")

        if os.path.exists("./output"):
            shutil.rmtree("./output")
        else:
            print("output file does not exist")

        os.mkdir("./process")
        os.mkdir("./input") 
        os.mkdir("./output")
    except Exception as e: print(e)
      
    if request.method == 'POST':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                save_location = os.path.join('input', filename)
                process_file = os.path.join('process', filename) 
                file.save(save_location)
                if filename.lower().endswith(('.pdf')):
                    pdf2img(save_location,filename.split('.')[0])
                elif filename.lower().endswith(('.png')):
                    png2jpg(save_location,filename.split('.')[0])
                else:
                    with Image.open(save_location) as image:
                        image.save(process_file, 'JPEG')
                new_filename = filename.split('.')[0]+'.jpg'
                process_file = os.path.join('process', new_filename) 

                #return send_from_directory('output', output_file) 
                
                
                with Image.open(process_file) as im:
                    width, height = im.size

                    sale_order = [[120*width/2616, (1127)*height/3385, 2552*width/2616, 1772*height/3385]
                        ,[140*width/2616, 441*height/3385, 2000*width/2616, 692*height/3385]
                        ,[140*width/2616, 731*height/3385, 600*width/2616, 963*height/3385]
                        ,[1653*width/2616,1811*height/3385,2600*width/2616, 2148*height/3385]]

                    invoice = [[120*width/2616, (1227)*height/3385, 2552*width/2616, 2372*height/3385]
                            ,[140*width/2616, 441*height/3385, 1650*width/2616, 865*height/3385]
                            ,[1650*width/2616, 441*height/3385, 2400*width/2616, 865*height/3385]
                            ,[1153*width/2616,2285*height/3385,2600*width/2616, 2480*height/3385]]

                    receipt = [[20*width/2616, (1120)*height/3385, 2552*width/2616, 2680*height/3385]
                            ,[140*width/2616, 441*height/3385, 2552*width/2616, 765*height/3385]
                            ,[1480*width/2616,2660*height/3385,2600*width/2616, 3000*height/3385]]
                  
                    file_path = os.path.join('process','information_0.jpg')
                    if request.form['format']=='sale_order':
                        crop_info(sale_order,process_file)
                        text=OCRextract(sale_order)
                        try:

                            df=order_extract(file_path,text)
                            
                            df.to_csv('./output/output.csv', index=False,encoding="utf-8")
                            df1=json_sale_order(text)
                            df1.to_json('./process/output.json',force_ascii=False, orient ='records')
                            with open("./process/table.json", encoding="utf-8") as json_file:
                                add_value = json.load(json_file)
                            with open("./process/output.json", encoding="utf-8") as json_file:
                                json_decoded = json.load(json_file)
                            json_decoded[0]['table'] = add_value
                            with open("./output/output.json", 'w', encoding="utf-8") as json_file:
                                json.dump(json_decoded, json_file,ensure_ascii=False) 
                        except Exception as e: print(e)
                        

                    elif request.form['format']=='invoice':
                        crop_info(invoice,process_file)
                        text=OCRextract(invoice)
                        try:
                            df=invoice_extract(file_path,text)
                            
                            df.to_csv('./output/output.csv', index=False,encoding="utf-8")
                            df1=json_invoice(text)
                            df1.to_json('./process/output.json',force_ascii=False, orient ='records')
                            with open("./process/table.json", encoding="utf-8") as json_file:
                                add_value = json.load(json_file)
                            with open("./process/output.json", encoding="utf-8") as json_file:
                                json_decoded = json.load(json_file)
                            json_decoded[0]['table'] = add_value
                            with open("./output/output.json", 'w', encoding="utf-8") as json_file:
                                json.dump(json_decoded, json_file,ensure_ascii=False) 
                        except Exception as e: print(e)

                    elif request.form['format']=='receipt':
                        crop_info(receipt,process_file)
                        text=OCRextract(receipt)
                        try:
                            df=receipt_extract(file_path,text)
                            
                            df.to_csv('./output/output.csv', index=False,encoding="utf-8")
                            df1=json_receipt(text)
                            df1.to_json('./process/output.json',force_ascii=False, orient ='records')
                            with open("./process/table.json", encoding="utf-8") as json_file:
                                add_value = json.load(json_file)
                            with open("./process/output.json", encoding="utf-8") as json_file:
                                json_decoded = json.load(json_file)
                            json_decoded[0]['table'] = add_value
                            with open("./output/output.json", 'w', encoding="utf-8") as json_file:
                                json.dump(json_decoded, json_file,ensure_ascii=False) 
                        except Exception as e: print(e)
                    else:
                        try:
                            df=t_extract(process_file)
                            
                            df.to_csv('./output/output.csv', index=False,encoding="utf-8")
                            df.to_json('./output/output.json',force_ascii=False, orient ='records')
                        except Exception as e: print(e)


                
                return redirect(url_for('download'))


    return render_template('index.html')
   
   
@app.route('/download')
def download():
    df = pd.DataFrame()
    try:
        df = pd.read_csv("./output/output.csv",encoding="utf-8")
    except Exception as e: print(e)
    return render_template('download.html', files=os.listdir('output'),tables=[df.to_html(index = False,classes='data', header="true")])

@app.route('/download/<filename>')
#@app.route('/<filename>')
def download_file(filename):
    return send_from_directory('output', filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
