from flask import Flask, render_template, request, jsonify, send_file, Response
import os
import glob
import cv2
import torchapp
from ultralytics import YOLO
import numpy as np
import io
import time
import threading

app = Flask(__name__)

# CUDA kontrolü
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA kullanılabilir. Şu anda kullanılan cihaz:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA kullanılabilir değil. CPU kullanılıyor.")

# Model yükleme
model = YOLO("pedestrianlight.pt")
model.to(device)

# Gerçek zamanlı video akışı için global değişkenler
camera = None
output_frame = None
lock = threading.Lock()

def get_next_output_filename(folder="videodetect", base_name="output_video", ext=".mp4"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    pattern = os.path.join(folder, f"{base_name}*{ext}")
    files = glob.glob(pattern)
    max_index = None
    for file in files:
        basename = os.path.basename(file)
        num_str = basename[len(base_name):-len(ext)]
        try:
            num = int(num_str)
            if max_index is None or num > max_index:
                max_index = num
        except ValueError:
            continue
    next_index = 1 if max_index is None else max_index + 1
    return os.path.join(folder, f"{base_name}{next_index}{ext}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/image')
def image_detection():
    return render_template('image.html')

@app.route('/video')
def video_detection():
    return render_template('video.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    confidence = float(request.form.get('confidence', 0.3))
    
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Resmi numpy dizisine çevir
        image_array = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Model tahmini
        results = model.predict(
            image,
            conf=confidence,
            imgsz=1280,
            device=device
        )
        
        # Sonucu görselleştir
        result_image = results[0].plot()
        
        # JPEG formatına dönüştür
        _, buffer = cv2.imencode('.jpg', result_image)
        
        # BytesIO nesnesine çevir
        io_buf = io.BytesIO(buffer)
        
        return send_file(
            io_buf,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='prediction.jpg'
        )

    except Exception as e:
        print(f"Hata: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'Video yüklenmedi'}), 400
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({'error': 'Video seçilmedi'}), 400
    
    try:
        # Geçici olarak videoyu kaydet
        temp_input = os.path.join('videodetect', 'temp_input.mp4')
        os.makedirs('videodetect', exist_ok=True)
        video_file.save(temp_input)

        # Video yakalayıcıyı başlat
        cap = cv2.VideoCapture(temp_input)
        
        if not cap.isOpened():
            return jsonify({'error': 'Video açılamadı'}), 400

        # Video özelliklerini al
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Çıktı dosya adını belirle
        output_path = get_next_output_filename(ext='.webm')  # WebM formatında kaydet
        output_filename = os.path.basename(output_path)

        # Video yazıcıyı ayarla - VP8 codec kullan
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Model ile tahmin yap
            results = model(frame, device=device)

            # Sonuçları işleyip çiz
            annotated_frame = results[0].plot()

            # İşlenmiş çerçeveyi dosyaya yaz
            out.write(annotated_frame)
            frame_count += 1

        # Kaynakları serbest bırak
        cap.release()
        out.release()

        # En az bir frame işlendiğinden emin ol
        if frame_count == 0:
            raise Exception("Video frames could not be processed")

        # Geçici dosyayı sil
        if os.path.exists(temp_input):
            os.remove(temp_input)

        print(f"Video başarıyla işlendi ve kaydedildi: {output_path}")
        print(f"Toplam işlenen frame sayısı: {frame_count}")

        # İşlenmiş video yolunu döndür
        return jsonify({
            'success': True,
            'video_path': output_filename,
            'message': 'Video başarıyla işlendi'
        })

    except Exception as e:
        print(f"Video işleme hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Temizlik işlemleri
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        if os.path.exists(temp_input):
            os.remove(temp_input)

@app.route('/video_result/<path:filename>')
def video_result(filename):
    try:
        # Videodetect klasöründen dosya yolunu oluştur
        video_path = os.path.join('videodetect', filename)
        
        # Dosyanın varlığını kontrol et
        if not os.path.exists(video_path):
            print(f"Dosya bulunamadı: {video_path}")
            return jsonify({'error': 'Video dosyası bulunamadı'}), 404

        return send_file(
            video_path,
            mimetype='video/webm',  # WebM formatı için MIME type
            as_attachment=False
        )
    except Exception as e:
        print(f"Video görüntüleme hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_video/<path:filename>')
def download_video(filename):
    try:
        # Videodetect klasöründen dosya yolunu oluştur
        video_path = os.path.join('videodetect', filename)
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video dosyası bulunamadı'}), 404
            
        return send_file(
            video_path,
            mimetype='video/webm',  # WebM formatı için MIME type
            as_attachment=True,
            download_name='processed_video.webm'  # .webm uzantısı ile indir
        )
    except Exception as e:
        print(f"Video indirme hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_latest_video')
def get_latest_video():
    try:
        # Videodetect klasöründeki tüm video dosyalarını bul
        video_files = glob.glob(os.path.join('videodetect', 'output_video*.webm'))  # WebM dosyalarını ara
        
        if not video_files:
            return jsonify({'error': 'Hiç video bulunamadı'}), 404
            
        # En son oluşturulan videoyu bul
        latest_video = max(video_files, key=os.path.getctime)
        video_filename = os.path.basename(latest_video)
        
        return jsonify({
            'success': True,
            'video_path': video_filename
        })
    except Exception as e:
        print(f"Son video bulma hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

def detect_realtime():
    global camera, output_frame, lock
    
    try:
        # Kamerayı başlat
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("Kamera açılamadı!")
            return
        
        # Kamera çözünürlüğünü ayarla
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        
        # FPS hesaplaması için değişkenler
        prev_frame_time = 0
        new_frame_time = 0
        
        while True:
            success, frame = camera.read()
            if not success:
                print("Frame okunamadı!")
                break
                
            try:
                # Frame'i döndür (kamera açısına göre)
                frame = cv2.flip(frame, 1)
                
                # FPS hesaplama
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                
                # YOLO ile tespit yap
                results = model(frame, device=device)
                
                # Sonuçları görselleştir
                annotated_frame = results[0].plot()
                
                # FPS bilgisini ekle
                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Frame'i güncelle
                with lock:
                    output_frame = annotated_frame.copy()
                    
            except Exception as e:
                print(f"Frame işleme hatası: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Kamera hatası: {str(e)}")
    finally:
        if camera is not None:
            camera.release()

def generate_frames():
    global output_frame, lock
    
    while True:
        with lock:
            if output_frame is None:
                continue
            
            try:
                # Frame'i JPEG formatına dönüştür
                _, buffer = cv2.imencode('.jpg', output_frame)
                frame_bytes = buffer.tobytes()
                
                # MIME formatında frame'i gönder
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Frame gönderme hatası: {str(e)}")
                continue
        
        # Küçük bir gecikme ekle
        time.sleep(0.01)

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream')
def start_stream():
    global camera, output_frame
    
    try:
        if camera is None:
            # Önceki frame'i temizle
            output_frame = None
            
            # Tespit thread'ini başlat
            t = threading.Thread(target=detect_realtime)
            t.daemon = True
            t.start()
            
            return jsonify({'success': True, 'message': 'Video akışı başlatıldı'})
        else:
            return jsonify({'success': False, 'message': 'Video akışı zaten çalışıyor'})
    except Exception as e:
        print(f"Stream başlatma hatası: {str(e)}")
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'})

@app.route('/stop_stream')
def stop_stream():
    global camera, output_frame
    
    if camera is not None:
        camera.release()
        camera = None
        output_frame = None
        return jsonify({'success': True, 'message': 'Video akışı durduruldu'})
    else:
        return jsonify({'success': False, 'message': 'Video akışı zaten durdurulmuş'})

if __name__ == '__main__':
    app.run(debug=True, threaded=True) 
    