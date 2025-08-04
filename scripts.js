document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const preview = document.getElementById("preview");
    const uploadBtn = document.getElementById("uploadBtn");

    // Preview uploaded image
    fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                preview.src = e.target.result;
                preview.style.display = "block";
            };
            reader.readAsDataURL(file);
        } else {
            preview.src = "";
            preview.style.display = "none";
        }
    });

    // Send image to FastAPI
    uploadBtn.addEventListener("click", async function () {
        const file = fileInput.files[0];
        if (!file) {
            alert("Please select a file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                document.getElementById("result").innerText = `Prediction: ${result.prediction} (${result.confidence})`;
                document.getElementById("details").innerText = JSON.stringify(result.probabilities, null, 2);
            } else {
                document.getElementById("result").innerText = `Error: ${result.error}`;
            }
        } catch (error) {
            document.getElementById("result").innerText = "Network error.";
            console.error(error);
        }
    });
});
