from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\kumax\.conda\envs\caps\Library\bin\tesseract.exe"
)
image = "1.PNG"
text = pytesseract.image_to_string(Image.open(image), lang="vie")
