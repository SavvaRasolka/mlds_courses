function drawLines() {
    const svg = document.getElementById('graph-lines');
    if (!svg) return;

    // Очищаем старые линии
    svg.innerHTML = '';

    const textCircle = document.querySelector('.text-circle');
    const fileCircle = document.querySelector('.file-circle');
    const submitCircle = document.querySelector('.submit-circle');
    const video = document.getElementById('video-container');

    if (!textCircle || !fileCircle || !submitCircle) return;


    function getCenter(el) {
        const rect = el.getBoundingClientRect();
        return {
            x: rect.left + rect.width / 2,
            y: rect.top + rect.height / 2
        };
    }

    const c1 = getCenter(textCircle);
    const c2 = getCenter(fileCircle);
    const c3 = getCenter(submitCircle);

    const pairs = [
        [c1, c3],
        [c2, c3],
    ];

    if (video.style.display === 'inline-block'){
        c4 = getCenter(video);
        pairs.push([c3, c4]);
        }

    

    const svgRect = svg.getBoundingClientRect();
    function toSvgCoords(point) {
        return {
            x: point.x - svgRect.left,
            y: point.y - svgRect.top
        };
    }

    pairs.forEach(([from, to]) => {
        const p1 = toSvgCoords(from);
        const p2 = toSvgCoords(to);
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', p1.x);
        line.setAttribute('y1', p1.y);
        line.setAttribute('x2', p2.x);
        line.setAttribute('y2', p2.y);
        // Можно добавить класс для стилизации
        line.classList.add('graph-line');
        svg.appendChild(line);
    });
}

// Запускаем при загрузке и при ресайзе
window.addEventListener('load', drawLines);
window.addEventListener('resize', drawLines);