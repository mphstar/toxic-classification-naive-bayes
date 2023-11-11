const input = document.getElementById("input");
const result = document.getElementById("hasil");

const proses = async () => {
  const res = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: JSON.stringify({
      text: input.value,
    }),
    headers: {
      "Content-Type": "application/json",
    },
  });

  const data = await res.json();

  let kontenHtml = `
                <strong>Result</strong><br>
                Toxic: ${data.result.toxic}<br>
                Severe Toxic: ${data.result.severe_toxic}<br>
                Obscene: ${data.result.obscene}<br>
                Threat: ${data.result.threat}<br>
                Insult: ${data.result.insult}<br>
                Identity Hate: ${data.result.identity_hate}<br>
            `;

  result.innerHTML = kontenHtml;

  if (input.value == "") result.innerHTML = "";
};
