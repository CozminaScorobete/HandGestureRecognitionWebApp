const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const outputCanvas = document.getElementById("outputCanvas");
const context = canvas.getContext("2d");
const outputContext = outputCanvas.getContext("2d");
const button_istoric = document.getElementById("istoric_scolar");
const button_note = document.getElementById("note");
const button_index = document.getElementById("index");
const customCursor = document.getElementById("customCursor");
var dropdown = document.getElementById("yearSelect");
const button_year = document.getElementById("yearFormButton");
let gest = null;

navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
    video.play();
  })
  .catch((err) => {
    console.error("Error accessing webcam: ", err);
  });

video.addEventListener("play", () => {
  function resizeCanvas() {
    outputCanvas.width = window.innerWidth;
    outputCanvas.height = window.innerHeight;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  setInterval(() => {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const frameData = canvas.toDataURL("image/jpeg");
    $.ajax({
      url: "/process_frame",
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify({ frameData: frameData }),
      success: (response) => {
        gest = response.gesture;

        const image = new Image();
        image.onload = () => {
          outputContext.save();
          outputContext.clearRect(
            0,
            0,
            outputCanvas.width,
            outputCanvas.height
          );
          outputContext.scale(-1, 1); // Inversează imaginea pe orizontală
          outputContext.drawImage(
            image,
            -outputCanvas.width,
            0,
            outputCanvas.width,
            outputCanvas.height
          );
          outputContext.restore();
        };
        image.src = response.frameData;

        if (response.indexFingerTipx && response.indexFingerTipy) {
          const adjustedX =
            ((canvas.width - response.indexFingerTipx) * outputCanvas.width) /
            canvas.width; // Inversăm coordonatele pe orizontală
          const adjustedY =
            (response.indexFingerTipy * outputCanvas.height) / canvas.height;
          customCursor.style.left = `${adjustedX}px`;
          customCursor.style.top = `${adjustedY}px`;

          const element = document.elementFromPoint(adjustedX, adjustedY);

          handleButtonHoverClick(button_index, adjustedX, adjustedY, gest);
          handleButtonHoverClick(button_istoric, adjustedX, adjustedY, gest);
          handleButtonHoverClick(button_note, adjustedX, adjustedY, gest);
          handleTableRowHoverClick(adjustedX, adjustedY, gest);
          handleDropdownHoverClick(dropdown, adjustedX, adjustedY, gest);
          handleButtonHoverClick(button_year, adjustedX, adjustedY, gest);
        }
      },
    });
  }, 100);
});

function handleButtonHoverClick(button, x, y, gesture) {
  const rect = button.getBoundingClientRect();
  const isHoveringButton =
    x >= rect.left &&
    x <= rect.right &&
    y >= rect.top &&
    y <= rect.bottom;

  if (isHoveringButton) {
    if (gesture === "Click") {
      button.click();
    } else if (gesture === "Hover") {
      button.style.backgroundColor = "#5f2c82";
      button.style.color = "#ffffff";
    }
  } else {
    if (gesture === "Hover") {
      button.style.backgroundColor = "";
      button.style.color = "";
    }
  }
}

function handleTableRowHoverClick(x, y, gesture) {
  const rows = document.querySelectorAll("table tbody tr");
  let isHoveringRow = false;
  let hoveredRow = null;

  rows.forEach((row) => {
    const rowRect = row.getBoundingClientRect();
    if (
      x >= rowRect.left &&
      x <= rowRect.right &&
      y >= rowRect.top &&
      y <= rowRect.bottom
    ) {
      isHoveringRow = true;
      hoveredRow = row;
    }
  });

  if (isHoveringRow) {
    if (gesture === "Hover") {
      hoveredRow.style.backgroundColor = "#5f2c82";
      hoveredRow.style.color = "#ffffff";
    }
  } else {
    rows.forEach((row) => {
      row.style.backgroundColor = "";
      row.style.color = "";
    });
  }
}

function handleDropdownHoverClick(dropdown, x, y, gesture) {
  const dropdownRect = dropdown.getBoundingClientRect();
  const isHoveringDropdown =
    x >= dropdownRect.left &&
    x <= dropdownRect.right &&
    y >= dropdownRect.top &&
    y <= dropdownRect.bottom;

  const options = dropdown.options;
  let isHoveringOption = false;
  let hoveredOption = null;
  for (let i = 0; i < options.length; i++) {
    const optionRect = options[i].getBoundingClientRect();
    if (x >= optionRect.left && x <= optionRect.right && y >= optionRect.top && y <= optionRect.bottom) {
      isHoveringOption = true;
      hoveredOption = options[i];
      break;
    }
  }

  if (isHoveringDropdown) {
    if (gesture === "Click") {
      dropdown.focus();
      dropdown.size = dropdown.options.length;
    } else if (gesture === "Hover") {
      dropdown.style.backgroundColor = "#5f2c82";
      dropdown.style.color = "#ffffff";
    }
  } else {
    if (gesture === "Hover") {
      dropdown.style.backgroundColor = "";
      dropdown.style.color = "";
    }
  }

  if (isHoveringOption) {
    if (gesture === "Click") {
      hoveredOption.selected = true;
    } else if (gesture === "Hover") {
      hoveredOption.style.backgroundColor = "#5f2c82";
      hoveredOption.style.color = "#ffffff";
    }
  } else {
    for (let i = 0; i < options.length; i++) {
      options[i].style.backgroundColor = "";
      options[i].style.color = "";
    }
  }
}
