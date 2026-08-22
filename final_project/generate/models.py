from django.db import models


class VideoTask(models.Model):
    task_id = models.CharField(max_length=255, unique=True, db_index=True)
    status = models.CharField(
        max_length=20,
        choices=[
            ('PENDING', 'Ожидает'),
            ('PROGRESS', 'В процессе'),
            ('COMPLETED', 'Завершено'),
            ('FAILED', 'Ошибка'),
        ],
        default='PENDING'
    )
    progress = models.PositiveSmallIntegerField(default=0)
    video_file = models.FileField(upload_to='videos/', null=True, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Task {self.task_id} - {self.status}"
