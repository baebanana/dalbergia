from django.shortcuts import render,redirect,get_object_or_404
from django.http import HttpResponse
from dataapp.models import Genus,Species,KnowledgeInfo,Image,AdminUser
from django.contrib import messages
from django.db.models import Q
from django.contrib.auth.hashers import check_password,make_password
import zipfile,io,os
import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image as PILImage
from django.core.files.storage import FileSystemStorage
from django.utils import timezone


# ==================================================
# ADMIN CHECK
# ==================================================

def admin_required(view_func):
    def _wrapped_view(request, *args, **kwargs):
        if 'admin_id' not in request.session:
            return redirect('formlogin')
        return view_func(request, *args, **kwargs)
    return _wrapped_view


# ==================================================
# BASIC PAGE
# ==================================================

def index(request):
    return render(request,"dataapp/homepage.html")

def home(request):
    knowledge_info = KnowledgeInfo.objects.all().order_by("info_creator")
    return render(request,"dataapp/homepage.html",{
        "knowledge": knowledge_info
    })

def managedata(request):
    return render(request,"dataapp/manage_data.html")

def managegenus(request):
    gn = Genus.objects.all()
    return render(request,"dataapp/manage_genus.html",{"genus":gn})

def managespeci(request):
    sp = Species.objects.all()
    return render(request,"dataapp/manage_species.html",{"spi":sp})

def manageinfo(request):
    kn = KnowledgeInfo.objects.all()
    return render(request,"dataapp/manage_info.html",{"kno":kn})


# ==================================================
# GENUS
# ==================================================

def addgenus(request):
    if request.method == "POST":
        Genus.objects.create(
            genus_name=request.POST["genus"],
            remarks=request.POST["remark"]
        )
        messages.success(request,"บันทึกข้อมูลเรียบร้อย")
        return redirect("genusdata")
    return render(request,"dataapp/genus_add.html")

def genusdelete(request,genu_id):
    Genus.objects.get(genus_id=genu_id).delete()
    messages.success(request,"ลบข้อมูลเรียบร้อย")
    return redirect("genusdata")

def genusupdate(request,gn_id):
    if request.method == "POST":
        gn = Genus.objects.get(genus_id=gn_id)
        gn.genus_name = request.POST["genusname"]
        gn.remarks = request.POST["remark"]
        gn.save()
        messages.success(request,"อัพเดตข้อมูลเรียบร้อย")
        return redirect("genusdata")

    gn = Genus.objects.get(genus_id=gn_id)
    return render(request,"dataapp/genus_update.html",{"gen":gn})

def genussearch(request):
    q = request.GET.get("name","")
    genus = Genus.objects.filter(genus_name__icontains=q)
    return render(request,"dataapp/genus_search.html",{
        "genus":genus,
        "name":q
    })


# ==================================================
# SPECIES
# ==================================================
def speciesdata(request):
    species_list = Species.objects.all()
    return render(request,"dataapp/species_page.html",{
        "species_list":species_list
    })

def addspecies(request):
    gn = Genus.objects.all()
    if request.method == "POST":
        Species.objects.create(
            sci_name=request.POST["sciname"],
            thai_name=request.POST["thainame"],
            description=request.POST["attri"],
            genus_id=request.POST["typegenus"]
        )
        messages.success(request,"บันทึกข้อมูลเรียบร้อย")
        return redirect("specidata")

    return render(request,"dataapp/species_add.html",{"genus":gn})

def deletespecies(request,spec_id):
    Species.objects.get(species_id=spec_id).delete()
    messages.success(request,"ลบข้อมูลเรียบร้อย")
    return redirect("specidata")

def updatespecies(request,spec_id):
    if request.method == "POST":
        spec = Species.objects.get(species_id=spec_id)
        spec.sci_name = request.POST["sciname"]
        spec.thai_name = request.POST["thainame"]
        spec.description = request.POST["descri"]
        spec.genus_id = request.POST["typegenus"]
        spec.save()
        messages.success(request,"อัพเดตข้อมูลเรียบร้อย")
        return redirect("specidata")

    speci = Species.objects.get(species_id=spec_id)
    genus = Genus.objects.all()
    return render(request,"dataapp/species_update.html",{"sp":speci,"gn":genus})

def searchspecies(request):
    query = request.GET.get("message","")
    results = Species.objects.select_related('genus').all()

    if query:
        results = results.filter(
            Q(thai_name__icontains=query)|
            Q(sci_name__icontains=query)|
            Q(description__icontains=query)|
            Q(genus__genus_name__icontains=query)
        )

    return render(request,"dataapp/species_search.html",{
        "species":results,
        "query":query
    })
    


# ==================================================
# ⭐ AI MODEL (ULTRA STABLE FOR RAILWAY)
# ==================================================

MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    'dataapp',
    'ml_models',
    'classify_plant_model.keras'
)

model = None

def get_model():
    global model
    if model is None:
        print("Loading AI model...")
        model = load_model(MODEL_PATH)
    return model


CLASS_NAMES = ['กระพี้นางนวล', 'พะยูง', 'เก็ดแดง', 'เครือคางควาย', 'เครือแมด']

def predictplant(request):
    result = None
    confidence = None
    image_url = None 
    if request.method == 'POST' and request.FILES.get('plant_image'):
        try:
            img_file = request.FILES['plant_image']

            # ✅ 1. บันทึกรูป
            fs = FileSystemStorage()
            filename = fs.save(img_file.name, img_file)
            image_url = fs.url(filename) # URL สำหรับแสดงผล

            # ✅ 2. โหลดรูปจากไฟล์ที่ save แล้ว
            img = PILImage.open(fs.path(filename)).convert("RGB")
            img = img.resize((224, 224))

            # ✅ 3. เตรียมรูปภาพ
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # 🚨 จุดสำคัญที่แก้: เอา preprocess_input ออก แล้วหาร 255.0 แทน
            img_array = img_array / 255.0

            # ✅ 4. ทำนายผล
            predictions = model.predict(img_array)
            result_index = int(np.argmax(predictions[0]))
            max_confidence = np.max(predictions[0]) # ดึงค่าความมั่นใจสูงสุด

            # 🚨 ระบบดักจับ Threshold 50% (ป้องกันรูปแอปเปิ้ล/กุ้ง)
            if max_confidence < 0.5:
                result = "ไม่สามารถระบุได้ (ภาพอาจไม่ใช่ใบไม้ หรือไม่ชัดเจน)"
                confidence = f"{max_confidence * 100:.2f}% (ความมั่นใจต่ำเกินไป)"
            else:
                if result_index < len(CLASS_NAMES):
                    result = CLASS_NAMES[result_index]
                    confidence = f"{max_confidence * 100:.2f}%"
                else:
                    result = "ไม่ทราบชนิด"
                    
        except Exception as e:
            print(e)
            result = "เกิดข้อผิดพลาดในการประมวลผลรูปภาพ"
            
    return render(request, 'dataapp/classify_page.html', {
            'result': result,
            'confidence': confidence,
            'image_url': image_url, # ส่งไป template
        })


# ==================================================
# LOGIN
# ==================================================

def login(request):
    return render(request,"dataapp/login_form.html")

def adminlogin(request):
    if request.method=="POST":
        user_in=request.POST.get('username')
        pass_in=request.POST.get('password')

        try:
            admin = AdminUser.objects.get(user_name=user_in)

            if pass_in == admin.password:
                request.session['admin_id']=admin.admin_id
                request.session['admin_name']=admin.full_name
                admin.last_login=timezone.now()
                admin.save()
                return redirect('mndata')
            else:
                messages.error(request,"รหัสผ่านไม่ถูกต้อง")

        except AdminUser.DoesNotExist:
            messages.error(request,"ไม่พบชื่อผู้ใช้งาน")

    return render(request,'dataapp/login_form.html')
