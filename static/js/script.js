const imageInput = document.getElementById("imageInput");
const imagePreviewContainer = document.getElementById("imagePreviewContainer");

imageInput.addEventListener("change", function (event) {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      const imgElement = document.createElement("img");
      imgElement.src = e.target.result;
      imgElement.classList.add("image-preview");
      imagePreviewContainer.innerHTML = "";
      imagePreviewContainer.appendChild(imgElement);
    };
    reader.readAsDataURL(file);
  }
});
