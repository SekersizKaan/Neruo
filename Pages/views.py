from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from PIL import Image
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np

# Roboflow modelini yükle
rf = Roboflow(api_key="bfnOdqFUJfxzZqOzqc2y")
project = rf.workspace().project("tumorneuro")
model = project.version(2).model

# Sınıf isimleri (doğrudan string eşleme)
CLASS_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary tumor": "Pituitary Tumor"
}

def convert_to_rgb_if_necessary(image_path):
    """Görseli RGBA ise RGB'ye dönüştür"""
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            rgb_img = img.convert('RGB')
            rgb_img.save(image_path)
            print(f"Görsel {image_path} RGBA'dan RGB'ye dönüştürüldü")
        return True
    except Exception as e:
        print(f"Görsel dönüştürme hatası: {str(e)}")
        return False

def process_image(image_path):
    """
    Yüklenen görseli işle ve Roboflow modelinden segmentasyon sonuçlarını döndür.
    """
    try:
        # Önce görsel formatını kontrol et ve gerekirse dönüştür
        if not convert_to_rgb_if_necessary(image_path):
            raise ValueError("Görsel formatı dönüştürülemedi")
        
        # Görseli OpenCV ile yükle
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Görsel yüklenemedi")
        
        # Model tahmini yap
        result = model.predict(image_path, confidence=90).json()
        
        # Supervision ile detections oluştur
        detections = sv.Detections.from_inference(result)
        
        # Etiketleri oluştur
        labels = []
        for item in result["predictions"]:
            class_name = item["class"].lower()  # Küçük harfe çevir
            display_name = CLASS_NAMES.get(class_name, class_name)
            confidence = item["confidence"]
            labels.append(f"{display_name} {confidence:.2f}")
        
        # Annotatörleri oluştur
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()
        
        # Segmentasyon maskelerini ve etiketleri ekle
        annotated_image = mask_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, 
            detections=detections, 
            labels=labels
        )
        
        # Sonuçları kaydet
        output_dir = 'static/detect_results'
        os.makedirs(output_dir, exist_ok=True)
        
        result_image_path = os.path.join(output_dir, 'result_image.jpg')
        cv2.imwrite(result_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        
        # Sonuçları işle
        output = []
        label_count = {}
        
        for item in result["predictions"]:
            class_name = item["class"].lower()
            display_name = CLASS_NAMES.get(class_name, class_name)
            confidence = item["confidence"]
            
            output.append({
                'label': display_name,
                'confidence': f"{confidence * 100:.2f}%",
                'box': {
                    'xmin': int(item['x'] - item['width'] / 2),
                    'ymin': int(item['y'] - item['height'] / 2),
                    'xmax': int(item['x'] + item['width'] / 2),
                    'ymax': int(item['y'] + item['height'] / 2)
                }
            })
            
            if display_name in label_count:
                label_count[display_name] += 1
            else:
                label_count[display_name] = 1
        
        return output, label_count, result_image_path
    
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        raise

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('image')
        if uploaded_file:
            try:
                # Görseli geçici bir dosyaya kaydetme
                temp_dir = 'temp'
                os.makedirs(temp_dir, exist_ok=True)
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_file_path, 'wb+') as temp_file:
                    for chunk in uploaded_file.chunks():
                        temp_file.write(chunk)

                # Model işlemine görseli gönderme
                result, label_count, result_image_path = process_image(temp_file_path)

                # Görselin URL'sini oluştur
                image_url = f'/static/detect_results/result_image.jpg'

                # Geçici dosyayı temizleme
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

                # Sonuçları ve görsel URL'sini render et
                return render(request, 'results.html', {
                    'result': result, 
                    'label_count': label_count, 
                    'image_url': image_url
                })
            
            except Exception as e:
                # Hata durumunda temizlik
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                return JsonResponse({
                    'success': False, 
                    'error': f"İşlem sırasında hata oluştu: {str(e)}"
                }, status=500)

    return JsonResponse({'success': False, 'error': 'Geçersiz istek!'}, status=400)
