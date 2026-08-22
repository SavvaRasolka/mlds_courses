from django.core.cache import cache
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import uuid
import os

from .forms import GenerateRequestForm 
from .tasks import generate_video_task


def index(request):
    form = GenerateRequestForm()
    return render(request, 'generate/index.html', {'form': form})


def generate(request):
    print(request.method)
    if request.method == 'POST':
        form = GenerateRequestForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            text = form.cleaned_data['text']

            ext = os.path.splitext(image.name)[1]
            filename = f"temp_{uuid.uuid4().hex}{ext}"
            file_path = default_storage.save(os.path.join('temp', filename), ContentFile(image.read()))
            full_path = default_storage.path(file_path)  # если нужно абсолютный путь

            task = generate_video_task.delay(full_path, text)
            

            return render(request, 'generate/index.html', {
                'form': form,          
                'task_id': task.id,
                'show_progress': True 
            })
