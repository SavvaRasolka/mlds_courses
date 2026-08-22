if (typeof taskId !== 'undefined' && taskId) {
    const submitCircle = document.querySelector('.submit-circle');
    const btn = document.getElementById('submit-btn');
    const spinner = submitCircle.querySelector('.spinner-border');
    
    btn.style.display = 'none';
    spinner.style.display = 'block';

    const interval = setInterval(async () => {
        try {
            const response = await fetch(`/celery-progress/${taskId}/`);
            const data = await response.json();
            const state = data.state;

            if (state === 'SUCCESS') {
                clearInterval(interval);
                spinner.style.display = 'none';
                btn.style.display = 'block';

                const videoUrl = data.result;
                if (videoUrl) {
                    const container = document.getElementById('video-container');
                    container.innerHTML = `
                        <video controls autoplay muted width="640">
                            <source src="${videoUrl}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    `;
                    container.style.display = 'inline-block';
                    if (typeof drawLines === 'function') {
                        drawLines();
                    }
                    const videoEl = container.querySelector('video');
                    if (videoEl) {
                        videoEl.addEventListener('loadeddata', function() {
                            if (typeof drawLines === 'function') {
                                drawLines();
                            }
                        });
                    }
                }
                
            } else if (state === 'FAILURE') {
                clearInterval(interval);
                spinner.style.display = 'none';
                btn.style.display = 'block';
                alert('Something went wrog');
            }
        } catch (error) {
            console.error('Task status error:', error);
        }
    }, 1000);
}