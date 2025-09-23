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
        td.textContent = row[h];
        tr.appendChild(td);
      });
      tableBody.appendChild(tr);
    
  });
  table.appendChild(tableBody);
  tableContainer.appendChild(table);
}

let categoryChartInstance;
let capacityChartInstance;
let claimProbChartInstance;

function renderCharts(summary){
    // Destroy old charts if they exist (prevents duplicates)
  if (categoryChartInstance) categoryChartInstance.destroy();
  if (capacityChartInstance) capacityChartInstance.destroy();
  if (claimProbChartInstance) claimProbChartInstance.destroy();

  const categoryLabels = summary.charts.category_pie.map(item => item.label);
  const categoryValues = summary.charts.category_pie.map(item => item.value);

  categoryChartInstance = new Chart(document.getElementById("categoryChart"), {
    type: "pie",
    data: {
      labels: categoryLabels,
      datasets: [{
        data: categoryValues,
        backgroundColor: ["#4caf50", "#ff9800", "#f44336"] // colors for Low/Med/High
      }]
    }
  })

  //CAPACITY BAR CHART
  const capacityLabels = summary.charts.capacity_bar.map(item => item.label);
  const capacityValues = summary.charts.capacity_bar.map(item => item.value);

  capacityChartInstance = new Chart(document.getElementById("capacityChart"), {
    type: "bar",
    data: {
      labels: capacityLabels,
      datasets: [{
        label: "Count",
        data: capacityValues,
        backgroundColor: "#2196f3"
      }]
    },
    options: { scales: { y: { beginAtZero: true } } }
  });
  
  //CLAIM PROBABILTY LINE CHART
  const probLabels = summary.charts.claim_prob_line.map(item => item.range);
  const probValues = summary.charts.claim_prob_line.map(item => item.count);

  claimProbChartInstance = new Chart(document.getElementById("claimProbChart"), {
      type: "line",
      data: {
        labels: probLabels,
        datasets: [{
          label: "Policies",
          data: probValues,
          fill: false,
          borderColor: "#9c27b0",
          tension: 0.3
        }]
      }
    });
}

//function for running scenarios through the backend endpoint
async function runScenario(adjustments) {
  const res = await fetch("http://localhost:8000/scenario-test/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ adjustments })
  });
  const result = await res.json();

  // Display comparison
  displayScenarioComparison(result.baseline, result.scenario);
}


function displaySummary(summary){
  const summaryContainer = document.getElementById("summaryContainer");
  summaryContainer.innerHTML = ""; //clear old

  if (!summary) {
    summaryContainer.innerHTML = "<p>No portfolio summary available.</p>";
    return;
  }

  const {
    totalPolicies,
    avgRiskScore,
    avgClaimProbability,
    totalExpectedLoss,
    categoryDistribution,
    capacityFlags
  } = summary;


  summaryContainer.innerHTML = `
  <h3>POTFOLIO SUMMARY</h3>
  <ul>
    <li><strong>Total Policies:</strong> ${totalPolicies}</li>
    <li><strong>Average Risk Score:</strong> ${(avgRiskScore * 100).toFixed(1)}%</li>
    <li><strong>Average Claim Probability:</strong> ${(avgClaimProbability * 100).toFixed(1)}%</li>
    <li><strong>Total Expected Loss:</strong> ${totalExpectedLoss.toLocaleString()}</li>
    <li><strong>Category Disctribution:</strong> ${JSON.stringify(categoryDistribution, null, 2)}</li>
    <li><strong>Capacity Flags:</strong> ${JSON.stringify(capacityFlags, null, 2)}</li>
  </ul>
  `
}

//get data from backend when you click upload button
uploadButton.addEventListener("click", async ()=>{
  const fileInput = document.getElementById("fileInput");
  const resultsDiv = document.getElementById("results");

  if (fileInput.files.length === 0) {
    resultsDiv.innerHTML = "<p style='color:red;'>Please select a CSV file first.</p>";
    return;
  }

  const file = fileInput.files[0];
  const fileName = fileInput.files[0].name;
  resultsDiv.innerHTML = `<p> File <strong>${fileName}</strong> uploaded successfully!</p>`;
 
  const formData = new FormData();
  formData.append('file', file);

  //Display the table before AI calculations
  Papa.parse(file, {
    header: true,
    dynamicTyping: true,
    complete: function(results) {
      console.log("Parsed CSV:", results.data);
      displayTable(results.data);
    }
  });

  //Fetch data from backend that includes AI manipulated table
  try{
    const response = await fetch("http://127.0.0.1:8000/upload-csv/", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    console.log("Backend response:", result);
    const {columns, rows}  = result;

    const table = document.createElement("table");
    table.classList.add("styled-table");
  
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    const prettyNames = {
      PolicyID: "Policy ID",
      Location: "Location",
      SumInsured: "Sum Insured",
      Premium: "Premium",
      ClaimsPaid: "Claims Paid",
      Construction: "Construction",
      YearBuilt: "Year Built",
      Occupancy: "Occupancy",
      predicted_risk: "Predicted Risk",
      category: "Risk Category",
      claim_probability: "Claim Probability",
      expected_loss: "Expected Loss",
      capacity_flag: "Capacity Flag"
    };

    columns.forEach((col)=> {
      const th = document.createElement("th");
      th.textContent = prettyNames[col] || col;
      headerRow.appendChild(th);
    })
    

    thead.appendChild(headerRow);
    table.appendChild(thead);

    //TABLE BODY
    const tbody = document.createElement("tbody");
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      columns.forEach((col)=>{
        const td = document.createElement("td");

        //Format the numbers nicely
        if (typeof row[col] === "number") {
          if (col === "predicted_risk" || col === "claim_probability") {
            td.textContent = (row[col] * 100).toFixed(1) + "%"; // convert to %
          } else if (col === "expected_loss" || col === "SumInsured" || col === "Premium" || col === "ClaimsPaid") {
            td.textContent = row[col].toLocaleString(); // add commas
          } else {
            td.textContent = row[col];
          }
        } else {
          td.textContent = row[col];
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    tableContainer.appendChild(table);

  displaySummary(result.summary);
  renderCharts(result.summary);

  }catch(error){
    console.error("Upload error", error);
    resultsDiv.innerHTML = `<p style='color:red;'>Error uploading file.</p>`;
  }

});


function displayScenarioComparison(baseline, scenario){
  scenarioResults.innerHTML = `
      <h3>Scenario Comparison</h3>
      <table class="styled-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Baseline</th>
            <th>Scenario</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Total Policies</td>
            <td>${baseline.totalPolicies}</td>
            <td>${scenario.totalPolicies}</td>
          </tr>
          <tr>
            <td>Average Risk Score</td>
            <td>${(baseline.avgRiskScore * 100).toFixed(1)}%</td>
            <td>${(scenario.avgRiskScore * 100).toFixed(1)}%</td>
          </tr>
          <tr>
            <td>Average Claim Probability</td>
            <td>${(baseline.avgClaimProbability * 100).toFixed(1)}%</td>
            <td>${(scenario.avgClaimProbability * 100).toFixed(1)}%</td>
          </tr>
          <tr>
            <td>Total Expected Loss</td>
            <td>${baseline.totalExpectedLoss.toLocaleString()}</td>
            <td>${scenario.totalExpectedLoss.toLocaleString()}</td>
          </tr>
        </tbody>
      </table>  
  `;
}

const scenarioForm = document.getElementById("scenarioForm");
const scenarioResults = document.getElementById("scenarioResults");

scenarioForm.addEventListener("submit", async (e) => {
  e.preventDefault();
    
  const column = document.getElementById("columnSelect").value;
  const condition = document.getElementById("conditionInput").value;
  const changePercent = parseFloat(document.getElementById("changeInput").value);

  if (!condition || isNaN(changePercent)) {
    scenarioResults.innerHTML = "<p style='color:red;'>Please enter valid condition and % change.</p>";
    return;
  }

  const changeDecimal = changePercent / 100;

  //build adjustment object
  const adjustments = {
    [column]: {
      [condition]: changeDecimal
    }
  };

  try{
  
    const res = await fetch("http://localhost:8000/scenario-test/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ adjustments })
    });

    const result = await res.json();
    if (result){
      console.log("WHAT IF RESULTS", result);
    }
    displayScenarioComparison(result.baseline, result.scenario);



  }catch(error){
    console.log("failed to run adjustments", error);
    scenarioResults.innerHTML = "<p style='color:red;'>Error running scenario.</p>";
  }

});