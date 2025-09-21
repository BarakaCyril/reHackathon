
const uploadButton = document.getElementById("uploadBtn");

uploadButton.addEventListener("click", ()=>{
  const fileInput = document.getElementById("fileInput");
  const resultsDiv = document.getElementById("results");

  if (fileInput.files.length === 0) {
    resultsDiv.innerHTML = "<p style='color:red;'>Please select a CSV file first.</p>";
    return;
  }

  const fileName = fileInput.files[0].name;
  resultsDiv.innerHTML = `<p>✅ File <strong>${fileName}</strong> uploaded successfully! (Processing coming soon...)</p>`;

});