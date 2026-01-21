
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
// const statusBox = document.getElementById("status");

// Asking for camera access
navigator.mediaDevices.getUserMedia({ video: true })
  .then(function(stream) {
    video.srcObject = stream;
  })
  .catch(function(err) {
    alert("Camera permission denied");
    console.error(err);
  });

//  Send frame every 100ms (10 FPS)
setInterval(function() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
  canvas.toBlob(function(blob) {
    const formData = new FormData();
    formData.append("frame", blob, "frame.jpg");
    fetch("/analyze_frame", {
      method: "POST",
      body: formData
    })
    .then(function(res) {
      return res.json();
    })
    .then(function(data) {
      console.log("Frame uploaded");
    })
    .catch(function(err) {
      console.error("Upload error:", err);
    });
  }, "image/jpeg", 0.7);
}, 100);

//  displaying the status of image on the website 
async function fetchStatus() {
  try {
    const res = await fetch("/status");
    const data = await res.json();
    document.getElementById("output").innerText = JSON.stringify(data, null, 2);
  } catch(err) {
    console.error("Status error:", err);
  }
}

setInterval(fetchStatus, 100);
fetchStatus();
 