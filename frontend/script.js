/* ===============================
   AI FRAUD COMMAND CENTER SCRIPT
   =============================== */

/* ---------- Generate PCA Inputs (V1–V28) ---------- */
const container = document.getElementById("features");

for (let i = 1; i <= 28; i++) {
    let input = document.createElement("input");
    input.type = "number";
    input.value = 0;
    input.placeholder = "V" + i;
    input.id = "V" + i;
    container.appendChild(input);
}


/* ---------- Chart Setup (Fraud Trend) ---------- */
const ctx = document.getElementById("fraudChart");

let fraudChart = new Chart(ctx, {
    type: "line",
    data: {
        labels: [],
        datasets: [{
            label: "Fraud Risk",
            data: [],
            borderColor: "#e50914",
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                labels: { color: "white" }
            }
        },
        scales: {
            x: {
                ticks: { color: "white" }
            },
            y: {
                ticks: { color: "white" },
                min: 0,
                max: 100
            }
        }
    }
});


/* ---------- MAIN PREDICTION FUNCTION ---------- */
async function predict() {

    try {

        /* Collect Inputs */
        let time = parseFloat(document.getElementById("time").value);
        let amount = parseFloat(document.getElementById("amount").value);

        let V_features = [];
        for (let i = 1; i <= 28; i++) {
            V_features.push(
                parseFloat(document.getElementById("V" + i).value)
            );
        }

        console.log("Sending prediction request...");

        /* API Call */
        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                time: time,
                amount: amount,
                V_features: V_features
            })
        });

        const result = await res.json();
        console.log("Backend response:", result);

        /* Calculate Risk */
        let risk = (result.fraud_probability * 100).toFixed(2);

        /* ---------- Update Text ---------- */
        document.getElementById("risk").innerText =
            "Risk Score: " + risk + "%";

        document.getElementById("status").innerText =
            result.prediction === 1
                ? "🚨 FRAUD DETECTED"
                : "✅ LEGITIMATE";

        /* ---------- Gauge Needle Animation ---------- */
        let angle = (risk / 100) * 180 - 90;
        const needle = document.getElementById("needle");

        needle.style.transition = "transform 1s ease";
        needle.style.transform = `rotate(${angle}deg)`;

        /* ---------- Progress Bar ---------- */
        const bar = document.getElementById("bar");
        bar.style.width = risk + "%";

        /* ---------- Alert Mode ---------- */
        if (risk > 80) {
            document.body.style.background = "#1a0000";

            new Audio(
                "https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg"
            ).play();
        } else {
            document.body.style.background = "#0b0f1a";
        }

        /* ---------- Update Chart ---------- */
        fraudChart.data.labels.push(
            new Date().toLocaleTimeString()
        );

        fraudChart.data.datasets[0].data.push(risk);

        if (fraudChart.data.labels.length > 10) {
            fraudChart.data.labels.shift();
            fraudChart.data.datasets[0].data.shift();
        }

        fraudChart.update();

        /* ---------- Add Transaction History ---------- */
        const historyTable = document.getElementById("history");

        let row = `
            <tr>
                <td>${new Date().toLocaleTimeString()}</td>
                <td>$${amount}</td>
                <td>${risk}%</td>
                <td>${result.prediction === 1 ? "FRAUD" : "SAFE"}</td>
            </tr>
        `;

        historyTable.innerHTML = row + historyTable.innerHTML;

    } catch (error) {

        console.error("Prediction Error:", error);

        document.getElementById("status").innerText =
            "❌ Backend Connection Failed";
    }
}