from PIL import Image
from transformers import DetrFeatureExtractor
from transformers import TableTransformerForObjectDetection
import torch
import numpy as np
import pandas as pd
import json
import easyocr
import re
import fitz  # PyMuPDF
from img2table.document import Image as Im
from img2table.ocr import EasyOCR
import os
import matplotlib.pyplot as plt


def pdf2img(pdf_path, output, dpi=300):
    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page
    #for page_num in range(len(pdf_document)):
        # Get the page
    page = pdf_document.load_page(0)

        # Render the page as an image with higher resolution
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))

        # Convert image to Pillow image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Save the image
    #img_path = f"process/{output}_{page_num+1}.jpg"
    img_path = f"process/{output}.jpg"
    img.save(img_path, dpi=(dpi, dpi))
    #print(f"Page {page_num + 1} saved as {img_path}")

    # Close the PDF
    pdf_document.close()

def png2jpg(png_file, jpg_file):
    try:
        # Open the PNG file
        with Image.open(png_file) as img:
            # Convert and save as JPG
            img.convert('RGB').save(f"process/{jpg_file}.jpg", 'JPEG')
        print(f"Conversion successful: {jpg_file}")
    except Exception as e:
        print(f"Conversion failed: {e}")

def strip_words(bounds):
  text=bounds
  for i in range(len(bounds)):
    text[i]=bounds[i].strip(" |")
  return text

def behind(found,text):
  for i in range(len(text)):
    if text[i] == found:
      info=text[i+1].strip(" |")
      return info
    
def between(before,text,after):
  for i in range(len(text)):
    if text[i] == before and text[i+2]!=after:
      info=text[i+1].strip(" |")
      return info
    
def find(pattern,text,define):

  #Will return all the strings that are matched

    for j in range(len(text)):
        if re.findall(pattern, text[j]):
            data=re.findall(pattern, text[j])
            if define == "date":
              patt = "\d{2}[/-]\d{2}[/-]\d{2}"
              data=re.findall(patt, text[j])
              for date in data:
                  day, month, year = map(int, date.split("/"))
                  if 1 <= day <= 31 and 1 <= month <= 12:
                    return date
            elif define == "tel":
              patt = "\d{2,3}[-]\d{3}[-]\d{4}"
              data=re.findall(patt, text[j])
              return data[0]

            else: return data[0]

def crop_info(form,image_path):
  with Image.open(image_path) as image:
  #image = Image.open(image_path)
    for i in range(len(form)):
      img_information = image.crop((form[i][0],form[i][1],form[i][2],form[i][3]))
      img_information.save("process/information_"+str(i)+".jpg")

def OCRextract(form):

  text=[]
  reader = easyocr.Reader(['th','en'],verbose=False,gpu=True)
  for i in range(1,len(form)):
    bounds = reader.readtext(os.path.join('process', 'information_'+str(i)+'.jpg'),paragraph=False, add_margin=0.13, slope_ths=1, height_ths=1, width_ths=1, detail=0)
    bounds = strip_words(bounds)
    #print(bounds)
    text.append(bounds)
  return text

#model = pickle.load(open("./models/tableDetector.pkl","rb"))
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
def compute_boxes(image_path):
      image = Image.open(image_path).convert("RGB")

      width, height = image.size
      feature_extractor = DetrFeatureExtractor()
      encoding = feature_extractor(image, return_tensors="pt")
      encoding.keys()
      
      with torch.no_grad():
          outputs = model(**encoding)

      results = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]
      boxes = results['boxes'].tolist()
      labels = results['labels'].tolist()
      plot_results(model,image, results['scores'], results['labels'], results['boxes'])

      return boxes,labels
def plot_results(model,pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=5,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    #plt.show()
    plt.savefig('./process/cropped.jpg', bbox_inches='tight')

def extract_table(image_path,header):
    empty_row=0
    reader = easyocr.Reader(['th','en'],gpu=True)
    with Image.open(image_path) as image:

      boxes,labels = compute_boxes(image_path)

      cell_locations = []

      for box_row, label_row in zip(boxes, labels):
          if label_row == 2:
              for box_col, label_col in zip(boxes, labels):
                  if label_col == 1:
                      cell_box = (box_col[0], box_row[1], box_col[2], box_row[3])
                      cell_locations.append(cell_box)

      cell_locations.sort(key=lambda x: (x[1], x[0]))

      num_columns = 0
      box_old = cell_locations[0]

      for box in cell_locations[1:]:
          x1, y1, x2, y2 = box
          x1_old, y1_old, x2_old, y2_old = box_old
          num_columns += 1
          if y1 > y1_old:
              break

          box_old = box

      df = pd.DataFrame(columns=header)

      row = []
      for box in cell_locations[:]:
          x1, y1, x2, y2 = box
          cell_image = image.crop((x1, y1, x2, y2))
          new_width = cell_image.width * 4
          new_height = cell_image.height * 4
          cell_image = cell_image.resize((new_width, new_height), resample=Image.LANCZOS)
          cell_image.save("./process/cell.jpg")
          cell_text = reader.readtext("./process/cell.jpg",paragraph=False, add_margin=0.13,
                                      slope_ths=1, height_ths=1, width_ths=1, detail=0)
          if len(cell_text) == 0:
            cell_text=np.nan
      
          else:
            cell_text = ''.join([str(elem) for elem in cell_text])
            #print(cell_text)

          row.append(cell_text)

          if len(row) == num_columns:
              if all(pd.isnull(row)):
                empty_row+=1
                if empty_row==2:
                  return df
              else: empty_row=0
              df.loc[len(df)] = row
              print(row)
              row = []

    return df

def order_extract(image_path,text):


      header = ['ลำดับ', 'รหัสสินค้า', 'รายการสินค้า', 'รายละเอียด/สี', 'XS', 'S', 'M', 'L', 'XL', 'จำนวน', 'ราคา', 'รวมราคา']
      df=extract_table(image_path,header)
      
      df.dropna(subset=['ลำดับ'], inplace=True)

      df.to_json('./process/table.json',force_ascii=False, orient ='records')
      df['ชื่อผู้สั่งซื้อ'] = behind('ชื่อผู้สั่งซื้อ',text[0])
      df['ที่อยู่'] = behind('ที่อยู่',text[0])
      df['โทร.'] = behind('โทร.',text[0])
      df['email'] = behind('email',text[0])
      df['วันที่'] = behind('วันที่',text[1])
      df['ชื่อแบรนด์'] = behind('ชื่อแบรนด์',text[1])
      df['วันนัดส่ง'] = behind('วันนัดส่ง',text[1])
      df['ราคา สินค้าก่อนหักภาษี ณ ที่จ่าย 3%'] = behind('ราคา สินค้าก่อนหักภาษี ณ ที่จ่าย 3%',text[2])
      df['ภาษี ณ ฺที่จ่าย 3%'] = behind('ภาษี ณ ฺที่จ่าย 3%',text[2])
      df['รวมราคาคงเหลือ'] = behind('รวมราคาคงเหลือ',text[2])
      df['ค่ามัดจำงาน'] = behind('ค่ามัดจำงาน',text[2])
      df['รวมสุทธิ'] = behind('รวมสุทธิ',text[2])
      df['ค่ามัดจำงานผลิต 70 %'] = behind('ค่ามัดจำงานผลิต 70 %',text[2])
      df['คงเหลือ'] = behind('คงเหลือ',text[2])


      return df

def invoice_extract(image_path,text):


      tel = '[tel,Tel,โทร] \d{2,3}[-]\d{3}[-]\d{4}'
      date = "วันที่ date \d{2}[/]\d{2}[/]\d{2}"
      Pay_d = "กำหนดชำระ \d{2}[/]\d{2}[/]\d{2}"
      header = ['item', 'Description', 'Quantity', 'Price','amount']
      df=extract_table(image_path,header)
      df.dropna(subset=["item"], inplace=True)
      df.to_json('./process/table.json',force_ascii=False, orient ='records')
      df['บริษัท']="บริษัท เวลเท็กซ์ เทรดดิ้ง จำกัด"
      df['Tel.'] = find(tel,text[0],"tel")
      df['วันที่'] = find(date,text[1],"date")
      df['กำหนดชำระ'] = find(Pay_d,text[1],"date")
      df['มูลค่าสินค้า'] = behind('มูลค่าสินค้า',text[2])
      df['จำนวนภาษีมูลค่าเพิ่ม'] = behind('จำนวนภาษีมูลค่าเพิ่ม',text[2])
      df['จำนวนเงินรวมทั้งสิ้น'] = behind('จำนวนเงินรวมทั้งสิ้น',text[2])

      return df

def receipt_extract(image_path,text):


      tel = '[tel,Tel,โทร] \d{2,3}[-]\d{3}[-]\d{4}'
      date = "\d{2}[-][^:|]+\s*[-]\d{2}"
      # Provide the path to your image file
      ocr = EasyOCR(lang=["en","th"])
      doc = Im(image_path)
      extracted_tables = doc.extract_tables(ocr=ocr,
                                            implicit_rows=False,
                                            borderless_tables=False,
                                            min_confidence=50)
      frames = []
      for table in extracted_tables:
          try:
              frames.append(table.df)
          except:
              frames.append(table)

      df = pd.concat(frames)
      df.columns = df.iloc[0]
      df = df[1:]
      df.to_json('./process/table.json',force_ascii=False, orient ='records')
      df['บริษัท']="บริษัท เวลเท็กซ์ เทรดดิ้ง จำกัด"
      df['โทร'] = find(tel,text[0],"tel")
      df['วันที่'] = find(date,text[1],"")
      df['รวมเงิน'] = between('รวมเงิน',text[1],'หักใบลดหนี้')
      df['หักใบลดหนี้'] = between('หักใบลดหนี้',text[1],'ยอดคงเหลือ')
      df['ยอดคงเหลือ'] = behind('ยอดคงเหลือ',text[1])

      return df
model2 = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
def t_extract(image_path):

      image=Image.open(image_path)
      width, height = image.size
      feature_extractor = DetrFeatureExtractor()
      encoding = feature_extractor(image, return_tensors="pt")
      encoding.keys()
      
      with torch.no_grad():
        outputs = model2(**encoding)
      results = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]
      im=image.crop((results['boxes'][0][0].tolist(),results['boxes'][0][1].tolist(),results['boxes'][0][2].tolist(),results['boxes'][0][3].tolist()))
      im.save("./process/cropped.jpg")
      ocr = EasyOCR(lang=["en","th"])
      doc = Im("./process/cropped.jpg")
      extracted_tables = doc.extract_tables(ocr=ocr,
                                            implicit_rows=False,
                                            borderless_tables=False,
                                            min_confidence=50)
      frames = []
      for table in extracted_tables:
          try:
              frames.append(table.df)
          except:
              frames.append(table)

      df = pd.concat(frames)
      df.columns = df.iloc[0]
      df = df[1:]
      df.to_json('./process/output.json',force_ascii=False, orient ='records')
   
      return df

def json_sale_order(text):

      information = np.array([[behind('ชื่อผู้สั่งซื้อ',text[0]), behind('ที่อยู่',text[0]),behind('โทร.',text[0]),behind('email',text[0])
                    , behind('วันที่',text[1]),behind('ชื่อแบรนด์',text[1]),behind('วันนัดส่ง',text[1])
                    ,behind('ราคา สินค้าก่อนหักภาษี ณ ที่จ่าย 3%',text[2]), behind('ภาษี ณ ฺที่จ่าย 3%',text[2]),behind('รวมราคาคงเหลือ',text[2])
                    ,behind('ค่ามัดจำงาน',text[2]),behind('รวมสุทธิ',text[2]),behind('ค่ามัดจำงานผลิต 70 %',text[2]),behind('คงเหลือ',text[2])]])

      # Convert data array into dataframe
      df = pd.DataFrame(information, columns = ['ชื่อผู้สั่งซื้อ', 'ที่อยู่', 'โทร.', 'email'
                      ,'วันที่', 'ชื่อแบรนด์','วันนัดส่ง'
                      ,'ราคา สินค้าก่อนหักภาษี ณ ที่จ่าย 3%', 'ภาษี ณ ฺที่จ่าย 3%%', 'รวมราคาคงเหลือ', 'ค่ามัดจำงาน', 'รวมสุทธิ', 'ค่ามัดจำงานผลิต 70 %', 'คงเหลือ'])

      df['table']=np.nan

    
      return df
def json_invoice(text):

    
      tel = '[tel,Tel,โทร] \d{2,3}[-]\d{3}[-]\d{4}'
      date = "วันที่ date \d{2}[/]\d{2}[/]\d{2}"
      Pay_d = "กำหนดชำระ \d{2}[/-]\d{2}[/-]\d{2}"
      information = np.array([["บริษัท เวลเท็กซ์ เทรดดิ้ง จำกัด", find(tel,text[0],"tel"),find(date,text[1],"date"),find(Pay_d,text[1],"date")
                    , behind('มูลค่าสินค้า',text[2]),behind('จำนวนภาษีมูลค่าเพิ่ม',text[2]),behind('จำนวนเงินรวมทั้งสิ้น',text[2])]])

      # Convert data array into dataframe
      df = pd.DataFrame(information, columns = ['บริษัท','Tel.','วันที่','กำหนดชำระ'
                      ,'มูลค่าสินค้า', 'จำนวนภาษีมูลค่าเพิ่ม', 'จำนวนเงินรวมทั้งสิ้น'])

      df['table']=np.nan

      return df

def json_receipt(text):

    
      tel = '[tel,Tel,โทร] \d{2,3}[-]\d{3}[-]\d{4}'
      date = "\d{2}[-][^:|]+\s*[-]\d{2}"
      information = np.array([["บริษัท เวลเท็กซ์ เทรดดิ้ง จำกัด", find(tel,text[0],"tel"),find(date,text[0],"")
                    ,between('รวมเงิน',text[1],'หักใบลดหนี้'),between('หักใบลดหนี้',text[1],'ยอดคงเหลือ')
                    ,behind('ยอดคงเหลือ',text[1])]])

      # Convert data array into dataframe
      df = pd.DataFrame(information, columns = ['บริษัท','โทร','วันที่','รวมเงิน'
                      ,'หักใบลดหนี้', 'ยอดคงเหลือ'])

      df['table']=np.nan

      return df


def json_table():
  with open("./process/table.json") as json_file:
    add_value = json.load(json_file)

  with open("./process/output.json") as json_file:
    json_decoded = json.load(json_file)

  json_decoded[0]['table'] = add_value
  with open("./process/output.json", 'w') as json_file:
      json.dump(json_decoded, json_file,ensure_ascii=False)
