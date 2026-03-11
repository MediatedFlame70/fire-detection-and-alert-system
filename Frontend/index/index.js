// FIRE CANVAS PARTICLE ANIMATION
const canvas = document.getElementById('fireCanvas');
const ctx = canvas.getContext('2d');
let W, H, particles = [];

function resize() {
  W = canvas.width = canvas.offsetWidth;
  H = canvas.height = canvas.offsetHeight;
}
resize();
window.addEventListener('resize', resize);

class Particle {
  constructor() { this.reset(true); }
  reset(initial) {
    this.x = Math.random() * W;
    this.y = initial ? Math.random() * H : H + 10;
    this.size = Math.random() * 5 + 1;
    this.speedY = Math.random() * 2.5 + 0.8;
    this.speedX = (Math.random() - 0.5) * 1.5;
    this.life = 1;
    this.decay = Math.random() * 0.012 + 0.004;
    this.hue = Math.random() * 45; // red to orange
  }
  update() {
    this.y -= this.speedY;
    this.x += this.speedX + Math.sin(this.y * 0.04) * 0.6;
    this.life -= this.decay;
    this.size *= 0.997;
    if (this.life <= 0) this.reset(false);
  }
  draw() {
    ctx.save();
    ctx.globalAlpha = this.life * 0.55;
    const g = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.size * 3.5);
    g.addColorStop(0, `hsl(${this.hue}, 100%, 70%)`);
    g.addColorStop(1, 'transparent');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size * 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

for (let i = 0; i < 140; i++) particles.push(new Particle());

function animate() {
  ctx.clearRect(0, 0, W, H);
  particles.forEach(p => { p.update(); p.draw(); });
  requestAnimationFrame(animate);
}
animate();

// SCROLL REVEAL ANIMATION
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry, i) => {
    if (entry.isIntersecting) {
      setTimeout(() => entry.target.classList.add('visible'), i * 80);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.step, .arch-block, .stat-card, .dataset-card, .team-card')
  .forEach(el => observer.observe(el));
