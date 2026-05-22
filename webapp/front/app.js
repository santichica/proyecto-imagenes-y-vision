const API_URL = "http://127.0.0.1:8000";

let chart = null;

// 🔥 GENERAR IMÁGENES
async function generate() {
    const num = document.getElementById("num").value;
    const container = document.getElementById("images");
    const loading = document.getElementById("loading");
    const button = document.querySelector("button");

    container.innerHTML = "";
    loading.classList.remove("hidden");
    button.disabled = true;

    try {
        const res = await fetch(`${API_URL}/generate?num_images=${num}`);
        const data = await res.json();

        loading.classList.add("hidden");

        data.images.forEach(path => {
            const img = document.createElement("img");
            img.src = `${API_URL}/${path}`;
            img.loading = "lazy";
            container.appendChild(img);
        });

    } catch (error) {
        loading.classList.add("hidden");
        alert("❌ Error generando imágenes");
        console.error(error);
    } finally {
        button.disabled = false;
    }
}

// 🔥 CAMBIO DE SECCIÓN
// 🔥 CAMBIO DE SECCIÓN (ACTUALIZADO PRO)
function showSection(sectionId) {

    // 🔥 ocultar secciones
    document.querySelectorAll(".section").forEach(sec => {
        sec.classList.remove("active");
    });

    // 🔥 mostrar sección actual
    document.getElementById(sectionId).classList.add("active");

    // 🔥 NAVBAR: estado activo bonito
    const navItems = document.querySelectorAll(".navbar li");

    navItems.forEach(li => {
        li.classList.remove("active");
    });

    // detectar cuál activar (más robusto)
    if (sectionId === "generator") {
        navItems[0].classList.add("active");
    } else if (sectionId === "results") {
        navItems[1].classList.add("active");
    }

    // 🔥 renderizar gráfica SOLO cuando se muestra
    if (sectionId === "results") {
        setTimeout(() => {
            createChart();
        }, 100);
    }
}

// 📊 CREAR GRÁFICA
function createChart() {
    const ctx = document.getElementById('fidChart');

    if (!ctx) return;

    // evitar duplicados
    if (chart) {
        chart.destroy();
    }

    const models = ["LoRA", "LoRA NEW", "img2img", "GAN", "GAN NEW"];

    const fidValues = [
        155.0089,
        155.0089,
        53.2477,
        254.8915,
        199.1471
    ];

    const colors = fidValues.map(v => {
        if (v < 80) return "#00ff99";   // bueno
        if (v < 150) return "#ffd166"; // medio
        return "#ef476f";              // malo
    });

    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [{
                label: 'FID (menor es mejor)',
                data: fidValues,
                backgroundColor: colors
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false, // 🔥 clave
            plugins: {
                legend: {
                    labels: {
                        color: "white"
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: "white" }
                },
                y: {
                    ticks: { color: "white" }
                }
            }
        }
    });
}