import Papa from "papaparse";

const uploadButton = document.getElementById("uploadBtn");
const tableContainer = document.getElementById("tableContainer");


function displayTable(data){
  tableContainer.innerHTML = "";


  if (data.length === 0) {
    tableContainer.innerHTML = "<p>No data found.</p>";
    return;
  }

  const table = document.createElement("table");

  //HEADERS
    const headers = Object.keys(data[0]); //extract keys in first row in the data i.e ID, Premiums
    const thead = document.createElement("thead"); // creates a table header section
    const headerRow = document.createElement("tr"); //first row inside the header
    headers.push("AI risk score");

    //loop through the keys in data to create table headers and append them in the first row
    headers.forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h;
      headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead)
  
  //ROWS
  const tableBody = document.createElement("tbody");
  data.forEach((row)=>{
     //creating a table row for each row in the data
    const tr = document.createElement("tr");
    //looping through the values in the header and not the actual header
    headers.forEach((h)=>{
      const td = document.createElement("td");

      if (h === "AI risk score"){
        td.textContent = "Medium"; //This is where the AI functionality will go
      }else{
        td.textContent = row[h];
      }
      tr.appendChild(td);
    });
    tableBody.appendChild(tr);
    
  });
  table.appendChild(tableBody);
  tableContainer.appendChild(table);



}

uploadButton.addEventListener("click", ()=>{
  const fileInput = document.getElementById("fileInput");
  const resultsDiv = document.getElementById("results");

  if (fileInput.files.length === 0) {
    resultsDiv.innerHTML = "<p style='color:red;'>Please select a CSV file first.</p>";
    return;
  }

  const file = fileInput.files[0];
  const fileName = fileInput.files[0].name;
  resultsDiv.innerHTML = `<p>✅ File <strong>${fileName}</strong> uploaded successfully!</p>`;

  Papa.parse(file, {
    header: true,
    dynamicTyping: true,
    complete: function(results) {
      console.log("Parsed CSV:", results.data);
      displayTable(results.data);
    }
  })

});