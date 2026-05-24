const API_URL = window.location.origin;

let chart = null;
let generatedImages = [];
let recallChart = null;

// GENERAR IMÁGENES
async function generate() {
    const num = document.getElementById("num").value;
    const container = document.getElementById("images");
    const loading = document.getElementById("loading");
    const button = document.querySelector(".controls button");
    const downloadBtn = document.getElementById("downloadBtn");

    container.innerHTML = "";
    loading.classList.remove("hidden");
    button.disabled = true;
    downloadBtn.classList.add("hidden");

    try {
        const res = await fetch(`${API_URL}/api/generate?num_images=${num}`);
        const data = await res.json();

        loading.classList.add("hidden");

        generatedImages = data.images; // 🔥 guardar rutas

        data.images.forEach(path => {
            const img = document.createElement("img");
            img.src = `${API_URL}/${path}`;
            img.loading = "lazy";
            container.appendChild(img);
        });

        // mostrar botón descarga
        downloadBtn.classList.remove("hidden");

    } catch (error) {
        loading.classList.add("hidden");
        alert("❌ Error generando imágenes");
        console.error(error);
    } finally {
        button.disabled = false;
    }
}

// DESCARGAR ZIP
async function downloadImages() {
    if (!generatedImages.length) return;

    const zip = new JSZip();

    const promises = generatedImages.map(async (path, index) => {
        const response = await fetch(`${API_URL}/${path}`);
        const blob = await response.blob();
        zip.file(`image_${index + 1}.png`, blob);
    });

    await Promise.all(promises);

    const content = await zip.generateAsync({ type: "blob" });

    const link = document.createElement("a");
    link.href = URL.createObjectURL(content);
    link.download = "imagenes.zip";
    link.click();
}

// CAMBIO DE SECCIÓN
function showSection(sectionId) {
    document.querySelectorAll(".section").forEach(sec => {
        sec.classList.remove("active");
    });

    document.getElementById(sectionId).classList.add("active");

    const navItems = document.querySelectorAll(".navbar li");
    navItems.forEach(li => li.classList.remove("active"));

    if (sectionId === "generator") navItems[0].classList.add("active");
    if (sectionId === "results") navItems[1].classList.add("active");

    if (sectionId === "results") {
        setTimeout(() => {
            createChart();
            createRecallChart();
        }, 100);
    }
}

// GRÁFICA
function createChart() {
    const ctx = document.getElementById('fidChart');

    if (!ctx) return;

    if (chart) chart.destroy();

    const models = ["LoRA", "LoRA NEW", "img2img", "GAN", "GAN NEW"];
    const fidValues = [155.0089, 155.0089, 53.2477, 254.8915, 199.1471];

    const colors = fidValues.map(v => {
        if (v < 80) return "#00ff99";
        if (v < 150) return "#ffd166";
        return "#ef476f";
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
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: "white" } }
            },
            scales: {
                x: { ticks: { color: "white" } },
                y: { ticks: { color: "white" } }
            }
        }
    });
}

// 🔥 CLASIFICAR IMAGEN
async function classifyImage() {
    console.log("FUNCION CARGADA");
    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Selecciona una imagen");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_URL}/api/derm/classify`, {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        const resultBox = document.getElementById("classificationResult");
        const label = document.getElementById("resultLabel");
        const barMel = document.getElementById("barMel");
        const barNevus = document.getElementById("barNevus");

        resultBox.classList.remove("hidden");

        const probMel = (data.prob_mel * 100).toFixed(2);
        const probNev = (data.prob_nv * 100).toFixed(2);

        label.innerText = data.label === "melanoma"
            ? `Posible Melanoma (${probMel}%)`
            : `Nevus (Benigno) (${probNev}%)`;

        barMel.style.width = `${data.prob_mel * 100}%`;
        barNevus.style.width = `${data.prob_nv * 100}%`;

    } catch (err) {
        console.error(err);
        alert("Error clasificando imagen");
    }
}

async function classifyImageReal() {
    console.log("FUNCION CARGADA");
    const fileInput = document.getElementById("imageInputReal");
    const file = fileInput.files[0];

    if (!file) {
        alert("Selecciona una imagen");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_URL}/api/real/classify`, {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        const resultBox = document.getElementById("classificationResult");
        const label = document.getElementById("resultLabel");
        const barMel = document.getElementById("barMel");
        const barNevus = document.getElementById("barNevus");

        resultBox.classList.remove("hidden");

        const probMel = (data.prob_mel * 100).toFixed(2);
        const probNev = (data.prob_nv * 100).toFixed(2);

        label.innerText = data.label === "melanoma"
            ? `Posible Melanoma (${probMel}%)`
            : `Nevus (Benigno) (${probNev}%)`;

        barMel.style.width = `${data.prob_mel * 100}%`;
        barNevus.style.width = `${data.prob_nv * 100}%`;

    } catch (err) {
        console.error(err);
        alert("Error clasificando imagen");
    }
}

async function classifyImageGAN() {
    console.log("FUNCION CARGADA");
    const fileInput = document.getElementById("imageInputGAN");
    const file = fileInput.files[0];

    if (!file) {
        alert("Selecciona una imagen");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_URL}/api/gan/classify`, {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        const resultBox = document.getElementById("classificationResult");
        const label = document.getElementById("resultLabel");
        const barMel = document.getElementById("barMel");
        const barNevus = document.getElementById("barNevus");

        resultBox.classList.remove("hidden");

        const probMel = (data.prob_mel * 100).toFixed(2);
        const probNev = (data.prob_nv * 100).toFixed(2);

        label.innerText = data.label === "melanoma"
            ? `Posible Melanoma (${probMel}%)`
            : `Nevus (Benigno) (${probNev}%)`;

        barMel.style.width = `${data.prob_mel * 100}%`;
        barNevus.style.width = `${data.prob_nv * 100}%`;

    } catch (err) {
        console.error(err);
        alert("Error clasificando imagen");
    }
}

function createRecallChart() {
    const ctx = document.getElementById('recallChart');

    if (!ctx) return;

    if (recallChart) recallChart.destroy();

    const labels = [
        "Baseline Real",
        "Híbrido TI",
        "Híbrido LoRA",
        "Híbrido GAN",
        "Derm s040",
        "Sint TI",
        "Sint LoRA",
        "Sint GAN",
        "Derm s005"
    ];

    const values = [
        0.528,
        0.585,
        0.535,
        0.579,
        0.604,
        0.000,
        0.006,
        0.006,
        0.478
    ];

    recallChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Recall Melanoma (mayor es mejor)',
                data: values,
                backgroundColor: values.map(v => {
                    if (v > 0.6) return "#00ff99";
                    if (v > 0.5) return "#ffd166";
                    return "#ef476f";
                })
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: "white" } }
            },
            scales: {
                x: { ticks: { color: "white" } },
                y: {
                    ticks: { color: "white" },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}