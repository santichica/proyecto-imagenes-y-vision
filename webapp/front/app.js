const API_URL = "http://127.0.0.1:8000";

async function generate() {
    const num = document.getElementById("num").value;
    const container = document.getElementById("images");
    const loading = document.getElementById("loading");
    const button = document.querySelector("button");

    // 🔥 Limpiar UI
    container.innerHTML = "";

    // 🔥 Mostrar loader
    loading.classList.remove("hidden");

    // 🔥 Deshabilitar botón
    button.disabled = true;

    try {
        const res = await fetch(`${API_URL}/generate?num_images=${num}`);
        const data = await res.json();

        // 🔥 Ocultar loader
        loading.classList.add("hidden");

        // 🔥 Mostrar imágenes
        data.images.forEach(path => {
            const img = document.createElement("img");
            img.src = `${API_URL}/${path}`;
            img.loading = "lazy";
            container.appendChild(img);
        });

    } catch (error) {
        // 🔥 Ocultar loader en error
        loading.classList.add("hidden");

        alert("❌ Error generando imágenes");
        console.error(error);
    } finally {
        // 🔥 Rehabilitar botón
        button.disabled = false;
    }
}

function showSection(sectionId) {
    document.querySelectorAll(".section").forEach(sec => {
        sec.classList.remove("active");
    });

    document.getElementById(sectionId).classList.add("active");
}