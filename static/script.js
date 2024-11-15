document.getElementById('prediction-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });
    const result = await response.text();
    document.getElementById('result').innerText = result;
});
